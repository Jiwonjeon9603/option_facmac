import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.option_facmac import ActFACMACDiscreteCritic, OptFACMACDiscreteCritic
from modules.discriminators.discri import Discrimination, MIOpt
# from components.action_selectors import multinomial_entropy
import torch as th
from torch.optim import RMSprop, Adam
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.qmix_ablations import VDNState, QMixerNonmonotonic
from utils.rl_utils import build_td_lambda_targets
from torch.distributions import Bernoulli, MultivariateNormal
import wandb
class OptionDiscreteLearner:
    def __init__(self, opt_mac, act_mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.opt_mac = opt_mac
        self.target_opt_mac = copy.deepcopy(self.opt_mac)
        if self.args.opt_parameter_sharing:
            self.opt_agent_params = list(opt_mac.parameters())
        else:
            self.opt_agent_params = list(opt_mac.parameters()[0])
            for i in range(self.n_agents-1):
                self.opt_agent_params += list(opt_mac.parameters()[i+1])

        self.act_mac = act_mac
        self.target_act_mac = copy.deepcopy(self.act_mac)
        if self.args.act_parameter_sharing:
            self.act_agent_params = list(act_mac.parameters())
        else:
            self.act_agent_params = list(act_mac.parameters()[0])
            for i in range(self.n_agents-1):
                self.act_agent_params += list(act_mac.parameters()[i+1])

        if self.args.use_discriminator:
            self.discri = Discrimination(scheme, args)
            self.discri_params = list(self.discri.parameters())
        if self.args.use_mi:
            self.miopt = MIOpt(scheme, args)
            self.discri_params += list(self.miopt.parameters())

        self.opt_critic = OptFACMACDiscreteCritic(scheme, args)
        self.target_opt_critic = copy.deepcopy(self.opt_critic)
        self.opt_critic_params = list(self.opt_critic.parameters())

        self.act_critic = ActFACMACDiscreteCritic(scheme, args)
        self.target_act_critic = copy.deepcopy(self.act_critic)
        self.act_critic_params = list(self.act_critic.parameters())

        self.opt_mixer = None
        if args.mixer is not None and self.args.n_agents > 1:  # if just 1 agent do not mix anything
            if args.mixer == "vdn":
                self.opt_mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.opt_mixer = QMixer(args)
            elif args.mixer == "vdn-s":
                self.opt_mixer = VDNState(args)
            elif args.mixer == "qmix-nonmonotonic":
                self.opt_mixer = QMixerNonmonotonic(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.opt_critic_params += list(self.opt_mixer.parameters())
            self.target_opt_mixer = copy.deepcopy(self.opt_mixer)


        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.opt_agent_optimiser = RMSprop(params=self.opt_agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
            self.act_agent_optimiser = RMSprop(params=self.act_agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
            self.opt_critic_optimiser = RMSprop(params=self.opt_critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)
            self.act_critic_optimiser = RMSprop(params=self.act_critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)
            self.discri_optimiser = RMSprop(params=self.discri_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.opt_agent_optimiser = Adam(params=self.opt_agent_params, lr=args.lr, eps=getattr(args, "optimizer_epsilon", 10E-8))
            self.act_agent_optimiser = Adam(params=self.act_agent_params, lr=args.lr, eps=getattr(args, "optimizer_epsilon", 10E-8))
            self.opt_critic_optimiser = Adam(params=self.opt_critic_params, lr=args.critic_lr, eps=getattr(args, "optimizer_epsilon", 10E-8))
            self.act_critic_optimiser = Adam(params=self.act_critic_params, lr=args.critic_lr, eps=getattr(args, "optimizer_epsilon", 10E-8))
            self.discri_optimiser = Adam(params=self.discri_params, lr=args.critic_lr, eps=getattr(args, "optimizer_epsilon", 10E-8))

        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.last_target_update_episode = 0
        self.critic_training_steps = 0

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        # actions = batch["actions"][:, :]
        actions = batch["actions_onehot"][:, :]
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]
        options = batch["options_onehot"][:, :]


        ### Train discriminators ###
        if self.args.use_discriminator:
            discri_inputs = self.build_discri_inputs(batch)
            log_discri_q = self.discri(discri_inputs)
            picked_discri_q = th.gather(log_discri_q, dim=-1, index=batch["actions"])
            discri_loss = -(picked_discri_q.squeeze()*mask).mean()

        if self.args.use_mi:
            var_opt = self.cal_mi(batch)
            picked_var_opt = th.gather(var_opt, dim=-1, index=batch["options"])
            mi_loss = -(picked_var_opt.squeeze()*mask).mean()

        tot_discri_loss = discri_loss + mi_loss
        self.discri_optimiser.zero_grad()
        tot_discri_loss.backward()
        discri_grad_norm = th.nn.utils.clip_grad_norm_(self.discri_params, self.args.grad_norm_clip)
        self.discri_optimiser.step()


        # Train the actor
        # Use gumbel softmax to reparameterize the stochastic policies as deterministic functions of independent
        # noise to compute the policy gradient (one hot action input to the critic)
        opt_mac_out = []
        act_mac_out = []
        opt_entropy_out = []
        act_entropy_out = []
        self.opt_mac.init_hidden(batch.batch_size)
        self.act_mac.init_hidden(batch.batch_size)
        
        zt = MultivariateNormal(th.zeros(self.args.common_latent_dim*batch.batch_size*batch.max_seq_length).cuda(), th.eye(self.args.common_latent_dim*batch.batch_size*batch.max_seq_length).cuda()).sample()
        zt = zt.reshape(batch.batch_size, batch.max_seq_length, self.args.common_latent_dim)

        for t in range(batch.max_seq_length):
            opt_act_outs = self.opt_mac.select_actions(batch, t_ep=t, t_env=t_env, test_mode=False, explore=False, zt=zt, zt_train=True)
            act_act_outs = self.act_mac.select_actions(batch, t_ep=t, t_env=t_env, test_mode=False, explore=False)
            opt_policy = self.opt_mac.forward(batch, t=t, return_logits=False, zt=zt, zt_train=True)
            act_policy = self.act_mac.forward(batch, t=t, return_logits=False)
            opt_entropy = -th.sum(opt_policy * th.log(opt_policy + 1e-10), dim=-1)
            act_entropy = -th.sum(act_policy * th.log(act_policy + 1e-10), dim=-1)
            opt_mac_out.append(opt_act_outs)
            act_mac_out.append(act_act_outs)
            opt_entropy_out.append(opt_entropy)
            act_entropy_out.append(act_entropy)
        
        opt_mac_out = th.stack(opt_mac_out, dim=1)  # Concat over time
        act_mac_out = th.stack(act_mac_out, dim=1)
        opt_entropy_out = th.stack(opt_entropy_out, dim=1)
        act_entropy_out = th.stack(act_entropy_out, dim=1)

        #### Calculate MI terms ####
        if self.args.use_discriminator:
            var_log_discri_q = self.discri(discri_inputs)
            discri_varq = th.sum(var_log_discri_q * act_mac_out, dim=-1)
            mi_discri = act_entropy_out + discri_varq
        
        if self.args.use_mi:
            var_var_opt = self.cal_mi(batch)
            chosen_varopt = th.sum(var_var_opt * opt_mac_out, dim=-1)
            mi_opt = opt_entropy_out + chosen_varopt

        chosen_option_qvals, _ = self.opt_critic(batch["obs"][:, :-1], opt_mac_out[:, :-1])
        chosen_action_qvals, _ = self.act_critic(th.cat([batch["obs"][:, :-1], options[:, :-1]], dim=-1), act_mac_out[:, :-1])
        #chosen_action_qvals, _ = self.critic(batch["obs"][:, :-1], mac_out)

        if self.opt_mixer is not None:
            if self.args.mixer == "vdn":
                chosen_option_qvals = self.opt_mixer(chosen_option_qvals.view(-1, self.n_agents, 1),
                                                 batch["state"][:, :-1])
                chosen_option_qvals = chosen_option_qvals.view(batch.batch_size, -1, 1)
            else:
                chosen_option_qvals = self.opt_mixer(chosen_option_qvals.view(batch.batch_size, -1, 1),
                                                 batch["state"][:, :-1])
        
        chosen_action_qvals = chosen_action_qvals.view(batch.batch_size, -1, self.n_agents)
        
        # Compute the actor loss
        opt_pg_loss = -((chosen_option_qvals - self.args.lam_mi * th.mean(mi_opt[:, :-1], dim=-1, keepdim=True)) * batch["high_termination"][:, :-1, :]).sum() / (batch["high_termination"][:, :-1, :].sum())
        act_pg_loss = -((chosen_action_qvals - self.args.lam_discri * mi_discri[:, :-1]) * mask[:, :-1]).sum()/ mask[:, :-1].sum()

        # Optimise agents
        self.opt_agent_optimiser.zero_grad()
        opt_pg_loss.backward()
        opt_agent_grad_norm = th.nn.utils.clip_grad_norm_(self.opt_agent_params, self.args.grad_norm_clip)
        self.opt_agent_optimiser.step()

        self.act_agent_optimiser.zero_grad()
        act_pg_loss.backward()
        act_agent_grad_norm = th.nn.utils.clip_grad_norm_(self.act_agent_params, self.args.grad_norm_clip)
        self.act_agent_optimiser.step()



########################################################### 0627까지 수정 함 이후 수정 필요 #####################################################

        # Train the critic batched
        target_opt_mac_out = []
        target_act_mac_out = []
        self.target_opt_mac.init_hidden(batch.batch_size)
        self.target_act_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_opt_act_outs = self.target_opt_mac.select_actions(batch, t_ep=t, t_env=t_env, test_mode=True, zt=zt, zt_train=True)
            target_act_act_outs = self.target_act_mac.select_actions(batch, t_ep=t, t_env=t_env, test_mode=True)
            target_opt_mac_out.append(target_opt_act_outs)
            target_act_mac_out.append(target_act_act_outs)
        target_opt_mac_out = th.stack(target_opt_mac_out, dim=1)  # Concat over time
        target_act_mac_out = th.stack(target_act_mac_out)

        opt_q_taken, _ = self.opt_critic(batch["obs"][:, :-1], options[:, :-1])
        act_q_taken, _ = self.act_critic(th.cat([batch["obs"][:, :-1], options[:, :-1]], dim=-1), actions[:, :-1])
        if self.opt_mixer is not None:
            if self.args.mixer == "vdn":
                opt_q_taken = self.opt_mixer(opt_q_taken.view(-1, self.n_agents, 1), batch["state"][:, :-1])
            else:
                opt_q_taken = self.opt_mixer(opt_q_taken.view(batch.batch_size, -1, 1), batch["state"][:, :-1])

        target_opt_vals, _ = self.target_opt_critic(batch["obs"][:, :], target_opt_mac_out.detach())
        target_act_vals, _ = self.target_act_critic(th.cat([batch["obs"][:, :], options[:, :]], dim=-1), target_act_mac_out.detach())
        
        if self.opt_mixer is not None:
            if self.args.mixer == "vdn":
                target_opt_vals = self.target_opt_mixer(target_opt_vals.view(-1, self.n_agents, 1), batch["state"][:, :])
            else:
                target_opt_vals = self.target_opt_mixer(target_opt_vals.view(batch.batch_size, -1, 1), batch["state"][:, :])

        if self.opt_mixer is not None:
            opt_q_taken = opt_q_taken.view(batch.batch_size, -1, 1)
            target_opt_vals = target_opt_vals.view(batch.batch_size, -1, 1)
        else:
            opt_q_taken = opt_q_taken.view(batch.batch_size, -1, self.n_agents)
            target_opt_vals = target_opt_vals.view(batch.batch_size, -1, self.n_agents)

        act_q_taken = act_q_taken.view(batch.batch_size, -1, self.n_agents)
        target_act_vals = target_act_vals.view(batch.batch_size, -1, self.n_agents)

        
        
        opt_targets = build_td_lambda_targets(batch["reward"], terminated, mask, target_opt_vals + self.args.lam_mi*th.mean(mi_opt, dim=-1, keepdim=True), self.n_agents,
                                          self.args.gamma, self.args.td_lambda)
        act_targets = build_td_lambda_targets(batch["reward"], terminated, mask, target_act_vals + self.args.lam_mi*mi_discri, self.n_agents,
                                          self.args.gamma, self.args.td_lambda)          
        
        opt_mask = batch["high_termination"][:, :-1]
        act_mask = mask[:, :-1]

        opt_td_error = (opt_q_taken - opt_targets.detach())
        opt_mask = act_mask.expand_as(opt_td_error)
        masked_opt_td_error = opt_td_error * opt_mask
        opt_loss = (masked_opt_td_error **2).sum() / opt_mask.sum()

        act_td_error = (act_q_taken - act_targets.detach())
        act_mask = act_mask.expand_as(act_td_error)
        masked_act_td_error = act_td_error * act_mask
        act_loss = (masked_act_td_error **2).sum() / act_mask.sum()

        self.opt_critic_optimiser.zero_grad()
        opt_loss.backward()
        opt_critic_grad_norm = th.nn.utils.clip_grad_norm_(self.opt_critic_params, self.args.grad_norm_clip)
        self.opt_critic_optimiser.step()

        self.act_critic_optimiser.zero_grad()
        act_loss.backward()
        act_critic_grad_norm = th.nn.utils.clip_grad_norm_(self.act_critic_params, self.args.grad_norm_clip)
        self.act_critic_optimiser.step()


        self.critic_training_steps += 1



        if getattr(self.args, "target_update_mode", "hard") == "hard":
            if (self.critic_training_steps - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
                self._update_targets()
                self.last_target_update_episode = self.critic_training_steps
        elif getattr(self.args, "target_update_mode", "hard") in ["soft", "exponential_moving_average"]:
            self._update_targets_soft(tau=getattr(self.args, "target_update_tau", 0.001))
        else:
            raise Exception(
                "unknown target update mode: {}!".format(getattr(self.args, "target_update_mode", "hard")))

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("opt_critic_loss", opt_loss.item(), t_env)
            self.logger.log_stat("act_critic_loss", act_loss.item(), t_env)
            self.logger.log_stat("opt_critic_grad_norm", opt_critic_grad_norm, t_env)
            self.logger.log_stat("act_critic_grad_norm", act_critic_grad_norm, t_env)
            wandb.log({"opt_critic_loss": opt_loss.item()}, step=t_env)
            wandb.log({"act_critic_loss": act_loss.item()}, step=t_env)
            wandb.log({"opt_pg_loss": opt_pg_loss.item()}, step=t_env)
            wandb.log({"act_pg_loss": act_pg_loss.item()}, step=t_env)
            wandb.log({"discri_loss": discri_loss.item()}, step=t_env)
            wandb.log({"mi_loss": mi_loss.item()}, step=t_env)
            wandb.log({"tot_discri_loss": tot_discri_loss.item()}, step=t_env)
            # mask_elems = mask.sum().item()
            # self.logger.log_stat("opt_td_error_abs", masked_opt_td_error.abs().sum().item() / mask_elems, t_env)
            # self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
            #                      t_env)
            self.log_stats_t = t_env



    def build_discri_inputs(self, batch):
        inputs = []
        inputs.append(batch["obs"])
        if self.args.obs_last_action:
            action_1 = th.zeros_like(batch["actions_onehot"][:, 0].unsqueeze(dim=1))
            action_2 = batch["actions_onehot"][:,:-1]
            concat_actions = th.concat([action_1, action_2], dim=1)
            inputs.append(concat_actions)
        inputs.append(th.eye(self.args.n_agents, device=batch.device).unsqueeze(0).expand(batch.batch_size, batch["obs"].shape[1], -1, -1))
        inputs.append(batch["options_onehot"])
        inputs = th.cat([x for x in inputs], dim=-1)

        return inputs

    def cal_mi(self, batch):
        var_opt = []
        for i in range(self.args.n_agents):
            q_var_opt = 0
            for j in range(self.args.n_agents):
                if j != i :
                    if self.args.obs_last_action and self.args.obs_agent_id:
                        agent_id_i = th.zeros(self.args.n_agents, device=batch.device)
                        agent_id_j = th.zeros(self.args.n_agents, device=batch.device)
                        agent_id_i[i] = 1
                        agent_id_j[j] = 1
                        agent_id_i = agent_id_i.unsqueeze(0).expand(batch.batch_size, batch["obs"].shape[1], -1)
                        agent_id_j = agent_id_j.unsqueeze(0).expand(batch.batch_size, batch["obs"].shape[1], -1)
                        first_action_i = th.zeros_like(batch["actions_onehot"][:, 0, i]).unsqueeze(dim=1)
                        first_action_j = th.zeros_like(batch["actions_onehot"][:, 0, j]).unsqueeze(dim=1)
                        last_action_i = th.cat([first_action_i, batch["actions_onehot"][:,:-1, i]], dim=1)
                        last_action_j = th.cat([first_action_j, batch["actions_onehot"][:,:-1, j]], dim=1)
                    
                        input_opt = th.cat([batch["obs"][:,:,i], last_action_i, agent_id_i, batch["obs"][:,:,j], last_action_j, agent_id_j, batch["options_onehot"][:,:,j]], dim=-1)


                    elif self.args.obs_agent_id:
                        agent_id_i = th.zeros(self.args.n_agents, device=batch.device)
                        agent_id_j = th.zeros(self.args.n_agents, device=batch.device)
                        agent_id_i[i] = 1
                        agent_id_j[j] = 1
                        agent_id_i = agent_id_i.unsqueeze(0).expand(batch.batch_size, batch["obs"].shape[1], -1)
                        agent_id_j = agent_id_j.unsqueeze(0).expand(batch.batch_size, batch["obs"].shape[1], -1)

                        input_opt = th.cat([batch["obs"][:,:,i], agent_id_i, batch["obs"][:,:,j], agent_id_j, batch["options_onehot"][:,:,j]], dim=-1)
                        
                    elif self.args.obs_last_action:
                        first_action_i = th.zeros_like(batch["actions_onehot"][:, 0, i]).unsqueeze(dim=1)
                        first_action_j = th.zeros_like(batch["actions_onehot"][:, 0, j]).unsqueeze(dim=1)
                        last_action_i = th.cat([first_action_i, batch["actions_onehot"][:,:-1, i]], dim=1)
                        last_action_j = th.cat([first_action_j, batch["actions_onehot"][:,:-1, j]], dim=1)
                    
                        input_opt = th.cat([batch["obs"][:,:,i], last_action_i, batch["obs"][:,:,j], last_action_j, batch["options_onehot"][:,:,j]], dim=-1)
                       
                    else:
                        input_opt = th.cat([batch["obs"][:,:,i], batch["obs"][:,:,j], batch["options_onehot"][:,:,j]], dim=-1)
                      
                    q_var_opt += self.miopt(input_opt)
                    
            var_opt.append(q_var_opt * 1/(self.args.n_agents -1))


        var_opt = th.stack(var_opt, dim=2)


        return var_opt


    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_opt_mac.parameters(), self.opt_mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_opt_critic.parameters(), self.opt_critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_act_mac.parameters(), self.act_mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_act_critic.parameters(), self.act_critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        if self.mixer is not None:
            for target_param, param in zip(self.target_opt_mixer.parameters(), self.opt_mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        if self.args.verbose:
            self.logger.console_logger.info("Updated all target networks (soft update tau={})".format(tau))

    def _update_targets(self):
        self.target_opt_mac.load_state(self.opt_mac)
        self.target_act_mac.load_state(self.act_mac)
        self.target_opt_critic.load_state_dict(self.opt_critic.state_dict())
        self.target_act_critic.load_state_dict(self.opt_critic.state_dict())
        if self.opt_mixer is not None:
            self.target_opt_mixer.load_state_dict(self.opt_mixer.state_dict())
        self.logger.console_logger.info("Updated all target networks")

    def cuda(self, device="cuda:0"):
        self.opt_mac.cuda(device=device)
        self.target_opt_mac.cuda(device=device)
        self.act_mac.cuda(device=device)
        self.target_act_mac.cuda(device=device)

        self.opt_critic.cuda(device=device)
        self.target_opt_critic.cuda(device=device)
        self.act_critic.cuda(device=device)
        self.target_act_critic.cuda(device=device)
        if self.opt_mixer is not None:
            self.opt_mixer.cuda(device=device)
            self.target_opt_mixer.cuda(device=device)
        if self.args.use_discriminator:
            self.discri.cuda(device=device)
        if self.args.use_mi:
            self.miopt.cuda(device=device)

    def save_models(self, path):
        self.opt_mac.save_models(path)
        self.act_mac.save_models(path)
        if self.mixer is not None:
            th.save(self.opt_mixer.state_dict(), "{}/opt_mixer.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.opt_mac.load_models(path)
        self.act_mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_opt_mac.load_models(path)
        self.target_act_mac.load_models(path)
        if self.mixer is not None:
            self.opt_mixer.load_state_dict(th.load("{}/opt_mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.agent_optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class OptMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY["gumbel"](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, explore=False, zt=None, zt_train=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        avail_options = th.ones_like(avail_actions)[:, :, :self.args.num_options]
        
        agent_outputs = self.forward(ep_batch, t_ep, return_logits=(not test_mode), zt=zt, zt_train=zt_train)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_options[bs], t_env,
                                                            test_mode=test_mode, explore=explore)
        if getattr(self.args, "use_ent_reg", False):
            return chosen_actions, agent_outputs
        return chosen_actions

    def forward(self, ep_batch, t, return_logits=True, zt=None, zt_train=False):
        agent_inputs = self._build_inputs(ep_batch, t, zt, zt_train)
        avail_actions = ep_batch["avail_actions"][:, t]
        
        if self.args.opt_parameter_sharing:
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        else:
            agent_outs = []
            for i in range(self.n_agents):
                agent_out, self.hidden_states = self.agents[i](agent_inputs[:,i].unsqueeze(dim=1), self.hidden_states)
                agent_outs.append(agent_out)
            agent_outs = th.cat(agent_outs, dim=0)
        if self.agent_output_type == "pi_logits":
            # if getattr(self.args, "mask_before_softmax", True):
            #     # Make the logits for unavailable actions very negative to minimise their affect on the softmax
            #     reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
            #     agent_outs[reshaped_avail_actions == 0] = -1e10
            if return_logits:
                return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        if self.args.opt_parameter_sharing:
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        else:
            for i in range(len(self.agents)):
                self.hidden_states = self.agents[i].init_hidden().unsqueeze(0).expand(batch_size, -1, -1)  # bav

    def parameters(self):
        if self.args.opt_parameter_sharing:
            return self.agent.parameters()
        else:
            return [self.agents[i].parameters() for i in range(len(self.agents))]

    def named_parameters(self):
        if self.args.opt_parameter_sharing:
            return self.agent.named_parameters()
        else:
            return [self.agents[i].named_parameters() for i in range(len(self.agents))]

    def load_state(self, other_mac):
        if self.args.opt_parameter_sharing:
            self.agent.load_state_dict(other_mac.agent.state_dict())
        else:
            for i in range(len(self.agents)):
                self.agents[i].load_state_dict(other_mac.agents[i].state_dict())

    def load_state_from_state_dict(self, state_dict):
        if self.args.opt_parameter_sharing:
            self.agent.load_state_dict(state_dict)
        else:
            for i in range(len(self.agents)):
                self.agents[i].load_state_dict(state_dict)

    def cuda(self, device="cuda"):
        if self.args.opt_parameter_sharing:
            self.agent.cuda(device=device)
        else:
            for i in range(len(self.agents)):
                self.agents[i].cuda(device=device)

    def _build_agents(self, input_shape):
        if self.args.opt_parameter_sharing:
            self.agent = agent_REGISTRY[self.args.opt_agent](input_shape, self.args)
        else:
            self.agents = [agent_REGISTRY[self.args.opt_agent](input_shape, self.args) for _ in range(self.args.n_agents)]
        #self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    # def share(self):
    #     self.agent.share_memory()

    def _build_inputs(self, batch, t, zt, zt_train):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        if self.args.use_common_latent_var:
            if zt_train:
                inputs.append(zt[:, t].unsqueeze(1).expand(-1, self.n_agents, -1))
            else:
                inputs.append(th.zeros(bs, self.args.n_agents, self.args.common_latent_dim, device=batch.device))

        try:
            inputs = th.cat([x for x in inputs], dim=-1)
        except Exception as e:
            pass
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        if self.args.use_common_latent_var:
            input_shape += self.args.common_latent_dim

        return input_shape

    def save_models(self, path):
        if self.args.opt_parameter_sharing:
            th.save(self.agent.state_dict(), "{}/opt_agent.th".format(path))
        else:
            for i in range(len(self.agents)):
                th.save(self.agents[i].state_dict(), "{}/opt_agent_"+str(i)+".th".format(path))

    def load_models(self, path):
        if self.args.opt_parameter_sharing:
            self.agent.load_state_dict(th.load("{}/opt_agent.th".format(path), map_location=lambda storage, loc: storage))
        else:
            for i in range(len(self.agents)): 
                self.agents[i].load_state_dict(th.load("{}/opt_agent_"+str(i)+".th".format(path), map_location=lambda storage, loc: storage))
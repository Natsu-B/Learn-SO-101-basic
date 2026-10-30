"""Validation-only controller: no learning and no target configuration access."""
import numpy as np

class DlsController:
    def __init__(self,sim,indices,max_delta=.05,damping=.01):
        self.sim=sim;self.indices=list(indices);self.max_delta=max_delta;self.damping=damping
        self.bounds=np.asarray(sim.action_bounds);self.command=np.asarray(sim.state()['position'])

    def action(self,target):
        state=self.sim.state();q=np.clip(state['position'],self.bounds[:,0],self.bounds[:,1])
        dq=np.asarray(self.sim.dls_delta(q.tolist(),list(target),self.indices,self.damping,self.max_delta))
        # DLS increments are based on the measured pose, not integrated targets.
        # Accumulation can excite the high-gain Menagerie servos.
        self.command=np.clip(q+dq,self.bounds[:,0],self.bounds[:,1])
        return self.command.tolist()

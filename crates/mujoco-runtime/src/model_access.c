#include <mujoco/mujoco.h>

/* rusty_mujoco 0.4.0's actuator_trnid getter compares a joint index to an
 * object-kind enum. Read the transmission using the pinned official headers. */
int robot_sim_actuator_joint(const mjModel* model, int actuator) {
  if (actuator < 0 || actuator >= model->nu) return -1;
  if (model->actuator_trntype[actuator] != mjTRN_JOINT) return -1;
  return model->actuator_trnid[2 * actuator];
}

/* Official 3.10 mjModel fields; only cached hinge position servos are accepted.
 * No topology, lengths, inertia, qpos0 or actuator types are changed. */
static int checked_dof(const mjModel* m, int a) {
  int j = robot_sim_actuator_joint(m, a);
  if (j < 0 || j >= m->njnt || m->jnt_type[j] != mjJNT_HINGE) return -1;
  int d = m->jnt_dofadr[j];
  return d >= 0 && d < m->nv ? d : -1;
}
int robot_sim_read_parameters(const mjModel* m, int a, double* out) {
  int d = checked_dof(m,a); if (d < 0) return 0;
  out[0]=m->dof_damping[d]; out[1]=m->dof_frictionloss[d];
  out[2]=m->actuator_gainprm[mjNGAIN*a];
  out[3]=m->actuator_biasprm[mjNBIAS*a+1];
  out[4]=m->actuator_biasprm[mjNBIAS*a+2];
  out[5]=m->actuator_forcerange[2*a];out[6]=m->actuator_forcerange[2*a+1];
  return 1;
}
int robot_sim_write_parameters(mjModel* m, int a, const double* values) {
  int d = checked_dof(m,a); if (d < 0) return 0;
  m->dof_damping[d]=values[0]; m->dof_frictionloss[d]=values[1];
  m->actuator_gainprm[mjNGAIN*a]=values[2];
  m->actuator_biasprm[mjNBIAS*a+1]=values[3];
  m->actuator_biasprm[mjNBIAS*a+2]=values[4];
  m->actuator_forcerange[2*a]=values[5];m->actuator_forcerange[2*a+1]=values[6];
  return 1;
}

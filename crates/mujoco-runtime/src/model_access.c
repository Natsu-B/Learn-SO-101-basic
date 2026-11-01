#include <mujoco/mujoco.h>
#include <math.h>

/* Only called on an owned rendering copy, never on a physics model. */
void robot_sim_render_quality(mjModel* model) {
  model->vis.quality.shadowsize = 1024;
  model->vis.quality.offsamples = 0;
}

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

/* Object (free body + one geom) physics for per-episode randomization.
 * Layout: mass, principal inertia[3], sliding friction, priority. */
static int checked_object(const mjModel* m, int b, int g) {
  return b > 0 && b < m->nbody && g >= 0 && g < m->ngeom && m->geom_bodyid[g] == b;
}
int robot_sim_read_object(const mjModel* m, int b, int g, double* out) {
  if (!checked_object(m, b, g)) return 0;
  out[0] = m->body_mass[b];
  for (int i = 0; i < 3; i++) out[1 + i] = m->body_inertia[3 * b + i];
  out[4] = m->geom_friction[3 * g];
  out[5] = m->geom_priority[g];
  return 1;
}
/* Writes only these fields, then mj_setConst recomputes the mass-dependent
 * constants (subtree mass, invweight0, stat.meaninertia, actuator_acc0) at
 * qpos0. The caller resets mjData afterwards; d is only scratch here. */
int robot_sim_write_object(mjModel* m, mjData* d, int b, int g, const double* v) {
  if (!checked_object(m, b, g) || !(v[0] > 0)) return 0;
  m->body_mass[b] = v[0];
  for (int i = 0; i < 3; i++) m->body_inertia[3 * b + i] = v[1 + i];
  m->geom_friction[3 * g] = v[4];
  m->geom_priority[g] = (int)v[5];
  mj_setConst(m, d);
  return 1;
}
int robot_sim_max_geom_priority(const mjModel* m) {
  int p = 0;
  for (int g = 0; g < m->ngeom; g++) if (m->geom_priority[g] > p) p = m->geom_priority[g];
  return p;
}

/* Contact fields of one geom. Layout: friction[3], condim, solref[2], priority.
 * Writing changes only these values (no sizes or topology); contacts created
 * after the write use them. */
int robot_sim_read_geom_contact(const mjModel* m, int g, double* out) {
  if (g < 0 || g >= m->ngeom) return 0;
  for (int i = 0; i < 3; i++) out[i] = m->geom_friction[3 * g + i];
  out[3] = m->geom_condim[g];
  out[4] = m->geom_solref[mjNREF * g];
  out[5] = m->geom_solref[mjNREF * g + 1];
  out[6] = m->geom_priority[g];
  return 1;
}
int robot_sim_write_geom_contact(mjModel* m, int g, const double* v) {
  if (g < 0 || g >= m->ngeom) return 0;
  int condim = (int)v[3];
  if (condim != 1 && condim != 3 && condim != 4 && condim != 6) return 0;
  for (int i = 0; i < 3; i++) m->geom_friction[3 * g + i] = v[i];
  m->geom_condim[g] = condim;
  m->geom_solref[mjNREF * g] = v[4];
  m->geom_solref[mjNREF * g + 1] = v[5];
  m->geom_priority[g] = (int)v[6];
  return 1;
}
const char* robot_sim_geom_name(const mjModel* m, int g) {
  const char* name = (g >= 0 && g < m->ngeom) ? mj_id2name(m, mjOBJ_GEOM, g) : NULL;
  return name ? name : "";
}
/* Solver options: timestep, integrator, solver, iterations, ls_iterations,
 * cone, impratio, noslip_iterations, tolerance, o_margin (override flag). */
void robot_sim_read_option(const mjModel* m, double* out) {
  out[0] = m->opt.timestep; out[1] = m->opt.integrator; out[2] = m->opt.solver;
  out[3] = m->opt.iterations; out[4] = m->opt.ls_iterations; out[5] = m->opt.cone;
  out[6] = m->opt.impratio; out[7] = m->opt.noslip_iterations; out[8] = m->opt.tolerance;
  out[9] = (m->opt.enableflags & mjENBL_OVERRIDE) ? 1 : 0;
}
/* Diagnostics: up to max contacts of geom g, ROBOT_SIM_CONTACT_FIELDS each:
 * other geom, other body, dist, dim, pos[3], normal into g[3], world force on
 * g[3], contact-frame force (normal, t1, t2), friction[0], solref[2],
 * solimp[0..2], frame tangent1 into g[3]. Returns the number written. */
#define ROBOT_SIM_CONTACT_FIELDS 27
int robot_sim_geom_contacts(const mjModel* m, const mjData* d, int g, double* out, int max) {
  int n = 0;
  for (int i = 0; i < d->ncon && n < max; i++) {
    const mjContact* c = d->contact + i;
    int other;
    double s;
    if (c->geom[1] == g) { other = c->geom[0]; s = 1; }       /* frame normal: geom1 -> geom2 */
    else if (c->geom[0] == g) { other = c->geom[1]; s = -1; }
    else continue;
    if (other < 0) continue;
    mjtNum f[6];
    mj_contactForce(m, d, i, f);
    double* o = out + ROBOT_SIM_CONTACT_FIELDS * n++;
    o[0] = other; o[1] = m->geom_bodyid[other]; o[2] = c->dist; o[3] = c->dim;
    for (int k = 0; k < 3; k++) {
      o[4 + k] = c->pos[k];
      o[7 + k] = s * c->frame[k];
      o[10 + k] = s * (f[0] * c->frame[k] + f[1] * c->frame[3 + k] + f[2] * c->frame[6 + k]);
      o[24 + k] = s * c->frame[3 + k];
    }
    o[13] = f[0]; o[14] = f[1]; o[15] = f[2];
    o[16] = c->friction[0]; o[17] = c->solref[0]; o[18] = c->solref[1];
    o[19] = c->solimp[0]; o[20] = c->solimp[1]; o[21] = c->solimp[2];
    o[22] = c->solimp[3]; o[23] = c->solimp[4];
  }
  return n;
}

/* Compliant-object edits for force_grasp.rs, reset-time only (the caller
 * then runs mj_setConst through robot_sim_write_object and resets mjData).
 * Box half sizes with the matching bounding sphere and local AABB. */
int robot_sim_set_box(mjModel* m, int g, const double* size) {
  if (g < 0 || g >= m->ngeom || m->geom_type[g] != mjGEOM_BOX) return 0;
  double r2 = 0;
  for (int i = 0; i < 3; i++) {
    if (!(size[i] > 0)) return 0;
    r2 += size[i] * size[i];
  }
  for (int i = 0; i < 3; i++) {
    m->geom_size[3 * g + i] = size[i];
    m->geom_aabb[6 * g + i] = 0;
    m->geom_aabb[6 * g + 3 + i] = size[i];
  }
  m->geom_rbound[g] = sqrt(r2);
  return 1;
}
int robot_sim_set_body_pos(mjModel* m, int b, const double* pos) {
  if (b <= 0 || b >= m->nbody) return 0;
  for (int i = 0; i < 3; i++) m->body_pos[3 * b + i] = pos[i];
  return 1;
}
/* Slide joint spring (N/m), damping (N s/m) and upper travel (m). */
int robot_sim_set_slide(mjModel* m, int j, double stiffness, double damping, double travel) {
  if (j < 0 || j >= m->njnt || m->jnt_type[j] != mjJNT_SLIDE) return 0;
  if (!(stiffness >= 0) || !(damping >= 0) || !(travel > 0)) return 0;
  m->jnt_stiffness[j] = stiffness;
  m->dof_damping[m->jnt_dofadr[j]] = damping;
  m->jnt_range[2 * j + 1] = travel;
  return 1;
}

"""Version-1/2 SI measurement interchange. Shared with Rust trajectory-io."""
from pathlib import Path
import csv
import json
import numpy as np


def header(names,version=1):
    if version==2:
        return ["measurement_time","command_time","applied_time","source_measurement_time"]+[f"{field}/{name}" for field in ("command","applied","position","velocity") for name in names]+["end_effector/x","end_effector/y","end_effector/z"]
    return ['timestamp']+[f'{field}/{name}' for field in ('command','position','velocity') for name in names]+['end_effector/x','end_effector/y','end_effector/z']


class Trajectory:
    def __init__(self, metadata, samples):
        self.metadata=dict(metadata)
        self.samples=np.asarray(samples,dtype=np.float64)
        names=metadata['joint_names']; n=len(names)
        if (metadata['schema_version'] not in (1,2) or metadata['backend'] not in ('mujoco','real') or not names
            or len(set(names))!=n or any(not name or '\n' in name or '\r' in name for name in names)
            or not np.isfinite(metadata['control_period']) or metadata['control_period']<=0
            or not np.isfinite(metadata['timestamp'])):
            raise ValueError('invalid trajectory metadata')
        for key in ('robot','model_version','seed','randomization_parameters'): metadata[key]
        dt=metadata['physics_timestep']
        if dt is not None and (not np.isfinite(dt) or dt<=0): raise ValueError('invalid physics timestep')
        base=1 if metadata["schema_version"]==1 else 4
        fields=3 if base==1 else 4
        if self.samples.ndim!=2 or self.samples.shape[1]!=base+fields*n+3 or len(self.samples)<1:
            raise ValueError('invalid sample dimensions or empty trajectory')
        if (not np.isfinite(self.samples[:,:base+(fields-1)*n]).all() or np.any(self.samples[:,0]<0)
            or np.any(np.diff(self.samples[:,0])<=0)):
            raise ValueError('invalid mandatory values or non-increasing timestamp')
        if metadata['schema_version']==2:
            if np.any(self.samples[:,1:4]>self.time[:,None]+1e-10) or np.any(self.samples[:,1:4]<0) or np.any(np.diff(self.command_time)<0):
                raise ValueError('invalid command/acquisition time')
        for block in (self.samples[:,base+(fields-1)*n:base+fields*n],self.samples[:,-3:]):
            if not np.all(np.all(np.isnan(block),axis=1)|np.all(np.isfinite(block),axis=1)):
                raise ValueError('optional sample fields must be all empty or all finite')

    @classmethod
    def from_csv(cls, directory):
        directory=Path(directory)
        metadata=json.loads((directory/'metadata.json').read_text())
        with (directory/'samples.csv').open(newline='') as f:
            reader=csv.reader(f)
            if next(reader)!=header(metadata['joint_names'],metadata['schema_version']): raise ValueError('CSV header mismatch')
            samples=[[float(x) if x else float('nan') for x in row] for row in reader]
        return cls(metadata,samples)

    def save(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        (directory/'metadata.json').write_text(json.dumps(self.metadata,indent=2,allow_nan=False)+'\n')
        with (directory/'samples.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(header(self.metadata['joint_names'],self.metadata['schema_version']))
            for row in self.samples: writer.writerow(['' if np.isnan(v) else repr(float(v)) for v in row])

    @property
    def time(self): return self.samples[:,0]
    @property
    def command_time(self): return self.time if self.metadata['schema_version']==1 else self.samples[:,1]
    @property
    def applied_time(self): return self.time if self.metadata['schema_version']==1 else self.samples[:,2]
    @property
    def command(self):
        n=len(self.metadata['joint_names']);b=1 if self.metadata['schema_version']==1 else 4
        return self.samples[:,b:b+n]
    @property
    def applied(self):
        n=len(self.metadata['joint_names']);return self.command if self.metadata['schema_version']==1 else self.samples[:,4+n:4+2*n]
    @property
    def position(self):
        n=len(self.metadata['joint_names']);b=1+n if self.metadata['schema_version']==1 else 4+2*n
        return self.samples[:,b:b+n]
    @property
    def velocity(self):
        n=len(self.metadata['joint_names']);b=1+2*n if self.metadata['schema_version']==1 else 4+3*n
        return self.samples[:,b:b+n]


def zero_order_hold(t, times, values):
    index=np.searchsorted(times,t,side='right')-1
    return values[np.maximum(index,0)]


def compare(sim, real, *, max_lag=.2, absolute_timestamps=False):
    """Positive lag means real follows sim. Metrics use overlapping timestamps, not indices.

    By default first sample times are subtracted separately (unsynchronized clocks).
    Lag uses mean-centered correlation on interpolated overlap, requires signal variance.
    This is a bounded estimate, not a SysID optimizer or clock synchronization service.
    """
    if sim.metadata['joint_names']!=real.metadata['joint_names']: raise ValueError('joint order mismatch')
    if len(sim.time)<8 or len(real.time)<8: raise ValueError('comparison requires >=8 samples per trace')
    if not np.isfinite(max_lag) or max_lag<0: raise ValueError('max_lag must be finite and nonnegative')
    st=sim.time.copy();rt=real.time.copy()
    if not absolute_timestamps: st-=st[0];rt-=rt[0]
    dt=max(float(np.median(np.diff(st))),float(np.median(np.diff(rt))))
    start=max(st[0],rt[0]);end=min(st[-1],rt[-1])
    grid=np.arange(start,end+dt*.01,dt)
    if len(grid)<8: raise ValueError('insufficient timestamp overlap')
    report={}
    for j,name in enumerate(sim.metadata['joint_names']):
        a=np.interp(grid,st,sim.position[:,j]);b=np.interp(grid,rt,real.position[:,j]);e=b-a
        metrics={'rmse_position':float(np.sqrt(np.mean(e*e))), 'mae_position':float(np.mean(abs(e))), 'maximum_error':float(np.max(abs(e)))}
        best=None
        for shift in np.arange(-int(max_lag/dt),int(max_lag/dt)+1)*dt:
            valid=(grid+shift>=rt[0])&(grid+shift<=rt[-1])
            if valid.sum()<max(8,len(grid)//2): continue
            x=a[valid];y=np.interp(grid[valid]+shift,rt,real.position[:,j])
            if x.std()<1e-9 or y.std()<1e-9: continue
            corr=float(np.corrcoef(x,y)[0,1]);score=(corr,-abs(shift))
            if best is None or score>best[0]: best=(score,float(shift),float(np.sqrt(np.mean((y-x)**2))))
        metrics['estimated_time_lag']=None if best is None else best[1]
        metrics['lag_corrected_rmse_position']=None if best is None else best[2]
        if np.isfinite(sim.velocity[:,j]).all() and np.isfinite(real.velocity[:,j]).all():
            ve=np.interp(grid,rt,real.velocity[:,j])-np.interp(grid,st,sim.velocity[:,j])
            metrics['rmse_velocity']=float(np.sqrt(np.mean(ve*ve)))
        report[name]=metrics
    return report,grid,st,rt


def log_comparison(sim, real, *, record=None, max_lag=.2, absolute_timestamps=False):
    import rerun as rr
    report,grid,st,rt=compare(sim,real,max_lag=max_lag,absolute_timestamps=absolute_timestamps)
    rr.init('sim-real-comparison')
    if record: rr.save(str(record))
    else: rr.spawn()
    for i,t in enumerate(grid):
        rr.set_time('elapsed_seconds',duration=float(t))
        rr.set_time('sample',sequence=i)
        for j,name in enumerate(sim.metadata['joint_names']):
            a=float(np.interp(t,st,sim.position[:,j]));b=float(np.interp(t,rt,real.position[:,j]))
            for path,value in [(f'sim/joints/{name}',a),(f'real/joints/{name}',b),(f'error/joints/{name}',b-a),
                               (f'sim/commands/{name}',float(zero_order_hold(t,sim.command_time-(0 if absolute_timestamps else sim.time[0]),sim.command[:,j]))),
                               (f'real/commands/{name}',float(zero_order_hold(t,real.command_time-(0 if absolute_timestamps else real.time[0]),real.command[:,j]))),
                               (f'sim/applied/{name}',float(zero_order_hold(t,sim.applied_time-(0 if absolute_timestamps else sim.time[0]),sim.applied[:,j]))),
                               (f'real/applied/{name}',float(zero_order_hold(t,real.applied_time-(0 if absolute_timestamps else real.time[0]),real.applied[:,j])))]:
                rr.log(path,rr.Scalars(value))
    rr.get_global_data_recording().flush()
    return report

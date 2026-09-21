"""Richer, non-redundant summaries from SD1.5 inverse trajectory tensors.

All features here are derived from latents/scores already produced by the
existing inversion calls. Scheduler transition residual is intentionally absent:
DDIMInverseScheduler explicitly constructs that transition from the same score,
so its residual would be tautological. Score change/alignment are the
non-tautological consistency measurements retained instead.
"""
from __future__ import annotations
import numpy as np
EPS=1e-8

def _flat(x): return np.asarray(x,dtype=np.float64).reshape(-1)
def _norm(x): return float(np.sqrt(np.mean(_flat(x)**2)))
def _cos(a,b):
    a=_flat(a);b=_flat(b);return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+EPS))
def _moments(x,prefix):
    a=_flat(x);m=float(a.mean());v=float(a.var());s=np.sqrt(v+EPS);q=np.quantile(a,[.05,.25,.5,.75,.95])
    return {f'{prefix}_mean':m,f'{prefix}_var':v,f'{prefix}_skew':float(np.mean(((a-m)/s)**3)),f'{prefix}_kurtosis':float(np.mean(((a-m)/s)**4)-3),f'{prefix}_q05':float(q[0]),f'{prefix}_q25':float(q[1]),f'{prefix}_median':float(q[2]),f'{prefix}_q75':float(q[3]),f'{prefix}_q95':float(q[4]),f'{prefix}_absmean':float(np.abs(a).mean()),f'{prefix}_rms':_norm(a),f'{prefix}_maxabs':float(np.abs(a).max()),f'{prefix}_nearzero':float(np.mean(np.abs(a)<.01))}
def _series(values,prefix):
    x=np.asarray(values,float);parts=np.array_split(x,3);out={f'{prefix}_t{i}':float(v) for i,v in enumerate(x)}
    out.update({f'{prefix}_mean':float(x.mean()),f'{prefix}_std':float(x.std()),f'{prefix}_min':float(x.min()),f'{prefix}_max':float(x.max()),f'{prefix}_early':float(parts[0].mean()),f'{prefix}_middle':float(parts[1].mean()),f'{prefix}_late':float(parts[-1].mean()),f'{prefix}_auc':float(np.trapezoid(x)/max(1,len(x)-1)),f'{prefix}_slope':float(np.polyfit(np.arange(len(x)),x,1)[0]) if len(x)>1 else 0.})
    return out
def _spatial(score,prefix):
    # score [C,H,W]; compact spatial/channel/frequency summaries.
    x=np.asarray(score,dtype=np.float64);c,h,w=x.shape; channel_rms=np.sqrt(np.mean(x*x,axis=(1,2)))
    out={f'{prefix}_channel_cv':float(channel_rms.std()/(channel_rms.mean()+EPS)),f'{prefix}_channel_ratio':float(channel_rms.max()/(channel_rms.min()+EPS)),f'{prefix}_channel_pr':float((channel_rms.sum()**2)/(np.square(channel_rms).sum()+EPS)/c),f'{prefix}_h_autocorr':_cos(x[:,:,:-1],x[:,:,1:]),f'{prefix}_v_autocorr':_cos(x[:,:-1,:],x[:,1:,:])}
    fft=np.fft.fft2(x,axes=(-2,-1));power=np.abs(fft)**2;fy=np.fft.fftfreq(h)[:,None];fx=np.fft.fftfreq(w)[None,:];radius=np.sqrt(fy*fy+fx*fx)
    total=float(power.sum())+EPS;low=float(power[:,radius<=.16].sum())/total;mid=float(power[:,(radius>.16)&(radius<=.33)].sum())/total;high=float(power[:,radius>.33].sum())/total
    p=(power/power.sum()).reshape(-1);out.update({f'{prefix}_fft_low':low,f'{prefix}_fft_mid':mid,f'{prefix}_fft_high':high,f'{prefix}_fft_high_low':high/(low+EPS),f'{prefix}_fft_entropy':float(-(p*np.log(p+EPS)).sum()/np.log(len(p))),f'{prefix}_fft_centroid':float((power*radius[None]).sum()/total)})
    return out
def _orth_fraction(a,b):
    cos=np.clip(_cos(a,b),-1,1);return float(np.sqrt(max(0.,1-cos*cos)))

def rich_features(result: dict) -> dict[str,float]:
    """Return rich features; requires capture_predictions=True on the probe."""
    if 'cond_scores' not in result: raise ValueError('rich features require captured conditional/unconditional scores')
    z=np.asarray(result['forward'],dtype=np.float64); ec=np.asarray(result['cond_scores'],dtype=np.float64); eu=np.asarray(result['uncond_scores'],dtype=np.float64); g=ec-eu; dz=np.diff(z,axis=0)
    out={}
    # Score distribution/spatial structure and latent typicality curves.
    for t in range(len(ec)):
        out.update(_moments(ec[t],f'rich_score_t{t}'));out.update(_moments(g[t],f'rich_guidance_t{t}'));out.update(_spatial(ec[t],f'rich_score_t{t}'))
        out.update(_moments(z[t],f'rich_latent_t{t}'))
        a=_flat(z[t]);out[f'rich_latent_t{t}_extreme3']=float(np.mean(np.abs(a)>3));out[f'rich_latent_t{t}_pr']=float((np.square(a).sum()**2)/(np.power(a,4).sum()+EPS)/len(a))
    # Score dynamics, including cross-time correlations.
    score_norm=[_norm(x) for x in ec];out.update(_series(score_norm,'rich_score_norm'))
    changes=[_norm(ec[t+1]-ec[t]) for t in range(len(ec)-1)];cos=[_cos(ec[t],ec[t+1]) for t in range(len(ec)-1)];angles=[float(np.arccos(np.clip(v,-1,1))) for v in cos]
    rel=[(score_norm[t+1]-score_norm[t])/(score_norm[t]+EPS) for t in range(len(score_norm)-1)];second=[_norm(ec[t+2]-2*ec[t+1]+ec[t]) for t in range(len(ec)-2)]
    out.update(_series(changes,'rich_score_change'));out.update(_series(cos,'rich_score_consecutive_cos'));out.update(_series(angles,'rich_score_angle'));out.update(_series(rel,'rich_score_relchange'));out.update(_series(second,'rich_score_secondchange'));out['rich_score_first_last_cos']=_cos(ec[0],ec[-1]);out['rich_score_cumulative_angle']=float(sum(angles))
    for i in range(len(ec)):
        for j in range(i+1,len(ec)):out[f'rich_score_crosscos_{i}_{j}']=_cos(ec[i],ec[j])
    # Latent movement / geometry.
    step=[_norm(v) for v in dz];out.update(_series(step,'rich_latent_step'));turn=[_cos(dz[t],dz[t+1]) for t in range(len(dz)-1)];turnang=[float(np.arccos(np.clip(v,-1,1))) for v in turn];acc=[_norm(dz[t+1]-dz[t]) for t in range(len(dz)-1)]
    out.update(_series(turn,'rich_latent_turn_cos'));out.update(_series(turnang,'rich_latent_turn_angle'));out.update(_series(acc,'rich_latent_accel'));path=sum(step);out['rich_latent_path_straightness']=_norm(z[-1]-z[0])/(path+EPS);out['rich_latent_endpoint_displacement']=_norm(z[-1]-z[0]);out['rich_latent_cumulative_turn']=float(sum(turnang))
    for i in range(len(dz)):
        for j in range(i+1,len(dz)):out[f'rich_dz_crosscos_{i}_{j}']=_cos(dz[i],dz[j])
    # Conditional/unconditional and score-trajectory alignment.
    align_c=[_cos(ec[t],dz[t]) for t in range(len(dz))];align_u=[_cos(eu[t],dz[t]) for t in range(len(dz))];align_g=[_cos(g[t],dz[t]) for t in range(len(dz))]
    for name,vals in [('cond_dz_align',align_c),('uncond_dz_align',align_u),('guidance_dz_align',align_g)]:out.update(_series(vals,f'rich_{name}'))
    out.update(_series([_orth_fraction(ec[t],dz[t]) for t in range(len(dz))],'rich_cond_dz_orthfrac'));out.update(_series([_orth_fraction(g[t],dz[t]) for t in range(len(dz))],'rich_guidance_dz_orthfrac'))
    out.update(_series([_cos(ec[t],eu[t]) for t in range(len(ec))],'rich_cond_uncond_cos'));out.update(_series([_norm(ec[t])/(_norm(eu[t])+EPS) for t in range(len(ec))],'rich_cond_uncond_normratio'));out.update(_series([_cos(g[t],ec[t]) for t in range(len(ec))],'rich_guidance_cond_align'))
    return out

from dataclasses import dataclass
from typing import Any
import jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict

@dataclass
class AnimaTextConditionerConfig:
  source_dim:int=1024; target_dim:int=1024; model_dim:int=1024; num_layers:int=6; num_attention_heads:int=16; mlp_ratio:float=4.0; target_vocab_size:int=32128; use_self_attention:bool=True; use_layer_norm:bool=False; min_sequence_length:int=512; dtype:Any=jnp.bfloat16; param_dtype:Any=jnp.bfloat16

def _rotate_half(x):
  h=x.shape[-1]//2; return jnp.concatenate((-x[...,h:],x[...,:h]),axis=-1)
def _rope(n,d,dtype):
  inv=1.0/(10000.0**(jnp.arange(0,d,2,dtype=jnp.float32)/d)); f=jnp.einsum('i,j->ij',jnp.arange(n,dtype=jnp.float32),inv); e=jnp.concatenate((f,f),axis=-1); return jnp.cos(e).astype(dtype),jnp.sin(e).astype(dtype)
class _RMSNorm(nn.Module):
  dim:int; eps:float=1e-6
  @nn.compact
  def __call__(self,x):
    w=self.param('weight',nn.initializers.ones,(self.dim,)); xf=x.astype(jnp.float32); y=xf/jnp.sqrt(jnp.mean(xf*xf,axis=-1,keepdims=True)+self.eps); return (y*w.astype(x.dtype)).astype(x.dtype)
class _Attention(nn.Module):
  query_dim:int; context_dim:int; heads:int; dtype:Any; param_dtype:Any
  @nn.compact
  def __call__(self,x,context=None,mask=None,apply_rope=True):
    context=x if context is None else context; d=self.query_dim//self.heads
    q=nn.Dense(self.heads*d,use_bias=False,dtype=self.dtype,param_dtype=self.param_dtype,name='q_proj')(x).reshape(x.shape[0],x.shape[1],self.heads,d)
    k=nn.Dense(self.heads*d,use_bias=False,dtype=self.dtype,param_dtype=self.param_dtype,name='k_proj')(context).reshape(context.shape[0],context.shape[1],self.heads,d)
    v=nn.Dense(self.heads*d,use_bias=False,dtype=self.dtype,param_dtype=self.param_dtype,name='v_proj')(context).reshape(context.shape[0],context.shape[1],self.heads,d)
    q=_RMSNorm(d,name='q_norm')(q); k=_RMSNorm(d,name='k_norm')(k)
    if apply_rope:
      qc,qs=_rope(q.shape[1],d,self.dtype); kc,ks=_rope(k.shape[1],d,self.dtype); q=q*qc[None,:,None,:]+_rotate_half(q)*qs[None,:,None,:]; k=k*kc[None,:,None,:]+_rotate_half(k)*ks[None,:,None,:]
    scores=jnp.einsum('bqhd,bkhd->bhqk',q.astype(jnp.float32),k.astype(jnp.float32))/(d**0.5)
    if mask is not None: scores=jnp.where(mask[:,None,None,:].astype(bool),scores,-1e4)
    y=jnp.einsum('bhqk,bkhd->bqhd',nn.softmax(scores,axis=-1).astype(self.dtype),v).reshape(x.shape[0],x.shape[1],-1)
    return nn.Dense(self.query_dim,use_bias=False,dtype=self.dtype,param_dtype=self.param_dtype,name='o_proj')(y)
class _Block(nn.Module):
  c:AnimaTextConditionerConfig
  @nn.compact
  def __call__(self,x,source,target_mask=None,source_mask=None):
    norm=lambda d,n: (_RMSNorm(d,name=n) if not self.c.use_layer_norm else nn.LayerNorm(name=n))
    if self.c.use_self_attention: x=x+_Attention(self.c.model_dim,self.c.model_dim,self.c.num_attention_heads,self.c.dtype,self.c.param_dtype,name='self_attn')(norm(self.c.model_dim,'norm_self_attn')(x),mask=target_mask)
    x=x+_Attention(self.c.model_dim,self.c.source_dim,self.c.num_attention_heads,self.c.dtype,self.c.param_dtype,name='cross_attn')(norm(self.c.model_dim,'norm_cross_attn')(x),context=source,mask=source_mask,apply_rope=True)
    h=nn.Dense(int(self.c.model_dim*self.c.mlp_ratio),use_bias=True,dtype=self.c.dtype,param_dtype=self.c.param_dtype,name='mlp_in')(norm(self.c.model_dim,'norm_mlp')(x)); h=nn.gelu(h,approximate=False); return x+nn.Dense(self.c.model_dim,use_bias=True,dtype=self.c.dtype,param_dtype=self.c.param_dtype,name='mlp_out')(h)
class FlaxAnimaTextConditioner(nn.Module):
  config:AnimaTextConditionerConfig
  @nn.compact
  def __call__(self,source_hidden_states,target_input_ids,source_attention_mask=None,target_attention_mask=None):
    c=self.config; x=nn.Embed(c.target_vocab_size,c.target_dim,dtype=c.dtype,param_dtype=c.param_dtype,name='embed')(target_input_ids)
    for i in range(c.num_layers): x=_Block(c,name=f'blocks_{i}')(x,source_hidden_states,target_attention_mask,source_attention_mask)
    x=nn.Dense(c.target_dim,use_bias=True,dtype=c.dtype,param_dtype=c.param_dtype,name='out_proj')(x); x=_RMSNorm(c.target_dim,name='norm')(x)
    if target_attention_mask is not None:
      x=x*target_attention_mask.astype(x.dtype)[...,None]
    pad=max(0,c.min_sequence_length-x.shape[1]); return jnp.pad(x,((0,0),(0,pad),(0,0)))[:,:c.min_sequence_length,:]
def load_and_convert_anima_text_conditioner_weights(path,params,dtype=jnp.bfloat16):
  from safetensors import safe_open
  flat=flatten_dict(params); out={}
  with safe_open(path,framework='pt',device='cpu') as f:
    def put(dst,src,tr=False):
      v=f.get_tensor(src).float().numpy(); v=v.T if tr else v; v=jnp.asarray(v,dtype=dtype)
      if tuple(v.shape)!=tuple(flat[dst].shape): raise ValueError(f'{src} {v.shape} != {dst} {flat[dst].shape}')
      out[dst]=v
    put(('embed','embedding'),'embed.weight'); put(('norm','weight'),'norm.weight'); put(('out_proj','kernel'),'out_proj.weight',True); put(('out_proj','bias'),'out_proj.bias')
    for i in range(6):
      p=f'blocks.{i}'; t=f'blocks_{i}'
      for a in ('self_attn','cross_attn'):
        for q in ('q_proj','k_proj','v_proj','o_proj'): put((t,a,q,'kernel'),f'{p}.{a}.{q}.weight',True)
        for n in ('q_norm','k_norm'): put((t,a,n,'weight'),f'{p}.{a}.{n}.weight')
      for n in ('norm_self_attn','norm_cross_attn','norm_mlp'): put((t,n,'weight'),f'{p}.{n}.weight')
      put((t,'mlp_in','kernel'),f'{p}.mlp.0.weight',True); put((t,'mlp_in','bias'),f'{p}.mlp.0.bias'); put((t,'mlp_out','kernel'),f'{p}.mlp.2.weight',True); put((t,'mlp_out','bias'),f'{p}.mlp.2.bias')
  return unflatten_dict(out)

"""Extract (action, state, instruction) per episode of libero_10_no_noops from the local RLDS copy (images skipped,
low memory) -> /tmp/libero10_actions.pkl.  Usage: python scripts/extract_demo_actions.py"""
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"; os.environ["CUDA_VISIBLE_DEVICES"]=""
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1); tf.config.threading.set_intra_op_parallelism_threads(1)
import tensorflow_datasets as tfds, numpy as np, pickle, sys
b=tfds.builder_from_directory('/root/robotic/bitvla-tar/data/modified_libero_rlds/libero_10_no_noops/1.0.0')
rc=tfds.ReadConfig(interleave_cycle_length=1, interleave_block_length=1, num_parallel_calls_for_interleave_files=1, num_parallel_calls_for_decode=1, shuffle_seed=0)
ds=b.as_dataset(split='train', decoders={'steps':{'observation':{'image':tfds.decode.SkipDecoding(),'wrist_image':tfds.decode.SkipDecoding()}}}, shuffle_files=False, read_config=rc)
eps=[]
for i,ep in enumerate(ds):
    A=[];S=[];instr=None
    for s in ep['steps']:
        A.append(s['action'].numpy()); S.append(s['observation']['state'].numpy())
        if instr is None: instr=s['language_instruction'].numpy().decode()
    eps.append(dict(action=np.stack(A),state=np.stack(S),instr=instr))
    if i%50==0: print(i,len(A),flush=True)
print(len(eps), sum(len(e['action']) for e in eps))
pickle.dump(eps,open('/tmp/libero10_actions.pkl','wb'))

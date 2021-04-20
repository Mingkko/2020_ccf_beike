import tensorflow as tf
import numpy as np


a = {
    'question_id':np.array([0,1,2,3,4]),
    'tokens':np.array([1,2,3,4,5])
}

ds = tf.data.Dataset.from_tensor_slices(a)
for x,y in ds:
    print(x,y)

print('done')
import time
import numpy as np
import tensorflow as tf
from progress.bar import Bar
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import Input



LAT_MAX = 48.99894
LAT_MIN = 24.36335
LON_MAX = -66.84510
LON_MIN = -125.12020
LAT_OR_LONG = 'LAT'
EPOCHS = 10
BATCH_SIZE = 100
BUFFER_SIZE = 500
NUM_SAMPLES = 50000
DME_BIN_COUNT = 10

MAE = tf.keras.losses.MeanAbsoluteError()
MSE = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()



def load_data():
    coord_filepath = r'E:\DATA\GEO-GUESS\aug-256\coordinates\coordinates_aug.txt'
    coord_lines = open(coord_filepath).readlines()
    bar = Bar(' LOADING DATA ', max = NUM_SAMPLES, suffix = '%(percent).1f%% - %(eta)ds')
    lat_array,long_array = [],[]
    for line in coord_lines[:NUM_SAMPLES]:
        path,lat,long,b = line.replace('\n','').split(',')
        s_lat,s_long = (float(lat) - LAT_MIN)/(LAT_MAX - LAT_MIN),(float(long) - LON_MIN)/(LON_MAX - LON_MIN)
        lat_array.append(s_lat)
        long_array.append(s_long)
        bar.next()
    bar.finish()
    if LAT_OR_LONG == 'LAT':
        train_dataset = tf.data.Dataset.from_tensor_slices(lat_array).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        return train_dataset
    elif LAT_OR_LONG == 'LONG':
        train_dataset = tf.data.Dataset.from_tensor_slices(long_array).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        return train_dataset
    else:
        print('TRAINING AXIS NOT RECOGNIZED')
        return None

def get_DME_A(input_values,output_values):
    #print(input_values.shape,output_values.shape)
    #s=input('..')
    pairs = np.stack((input_values,output_values),axis=1)
    output_sorted = pairs[pairs[:,0].argsort()][:,1]
    consec_output_pairs = np.stack((output_sorted[:-1],output_sorted[1:]),axis=1)
    comparison_array_sum = (consec_output_pairs[:,0] < consec_output_pairs[:,1]).sum()
    

    return ((output_values.size - 1) - comparison_array_sum)/(output_values.size - 1)
    

def get_DME_A(input_values,output_values):
    pairs = tf.stack((input_values,output_values),axis=1)
    output_sorted = tf.argsort(pairs[pairs[:,0]])[:,1]

'''
def get_DME_B(input_values,output_values,bin_count):
    bins = np.array([n/bin_count for n in range(bin_count+1)])
    dist = np.histogram(output_values,bins)[0]/output_values.size
    target_array = np.full(bin_count,output_values.size/bins.size)/output_values.size
    
    return MAE(dist,target_array).numpy()
'''

#def get_DME_B(input_values,output_values,bin_count):
    

@tf.custom_gradient
def DME(input_values,output_values,bin_count):


    output_values = tf.squeeze(output_values)

    #DME_A = get_DME_A(input_values,output_values)

    # GET B
    dist = tf.histogram_fixed_width(output_values,[0.0,1.0],nbins=DME_BIN_COUNT)/tf.size(output_values)
    target_array = tf.fill((DME_BIN_COUNT,),tf.size(output_values)/DME_BIN_COUNT)/tf.cast(tf.size(output_values),dtype=tf.float64)
    return MAE(dist,target_array)

    #DME_B = get_DME_B(input_values,output_values,bin_count)
    #return DME_B


def get_model():

    model = Sequential()

    model.add(Input(shape=(1,)))

    model.add(Dense(10,activation='sigmoid',kernel_initializer='he_uniform'))

    model.add(Dense(15,activation='sigmoid',kernel_initializer='he_uniform'))
    #model.add(Dense(1,activation='sigmoid',kernel_initializer='he_uniform'))
    model.add(Dense(10,activation='sigmoid',kernel_initializer='he_uniform'))

    model.add(Dense(1,kernel_initializer='he_uniform',activation='sigmoid'))

    model.compile(loss=DME)
    
    return model

def train(train_dataset,model):
    for epoch in range(EPOCHS):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        for step, x_batch in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                x_batch = tf.sort(x_batch)
                logits = model(x_batch, training=True)
                #loss_value = DME(x_batch, logits, DME_BIN_COUNT)
                loss_value = MSE(x_batch,logits)
                #print("XBATCH")
                #print(x_batch)
                #print("YBATCH")
                #print(logits)

            #vars = tf.trainable_variables()

            
            grads = tape.gradient(loss_value, model.trainable_weights)

            #print(grads)
            
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            #s = input('..')

            # Log every 200 batches.
            if step % 10 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %d samples" % ((step + 1) * BATCH_SIZE))



def main():

    train_data = load_data()

    model = get_model()

    train(train_data,model)



if __name__ == '__main__':
    main()
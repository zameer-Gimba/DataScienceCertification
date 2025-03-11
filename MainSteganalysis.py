import tensorflow as tf

def HybridizeLUActivationFunctionInCNNForSteganalysis(model):
    activation_functions = ['elu', 'relu', 'leakyrelu']
    modified_layers = []
    
    for i in range(5):
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Activation):
                activation_name = layer.get_config()['activation']
                activation_output = layer(layer.input)
                
                if tf.reduce_sum(activation_output) < 0:
                    new_activation = tf.keras.layers.ELU()
                elif tf.reduce_sum(activation_output) > 0:
                    new_activation = tf.keras.layers.ReLU()
                else:
                    new_activation = tf.keras.layers.LeakyReLU()
                
                modified_layers.append(new_activation)
            else:
                modified_layers.append(layer)
        
        model = tf.keras.models.Sequential(modified_layers)
        modified_layers = []
    
    return model

# Create a sample model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Apply the function to modify the model
modified_model = HybridizeLUActivationFunctionInCNNForSteganalysis(model)

# Print the modified model summary
modified_model.summary()

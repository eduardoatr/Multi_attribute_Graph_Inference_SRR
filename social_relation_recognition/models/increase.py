import tensorflow as tf


def get_model_edge_mlp(
    features_list,
    feature2size,
    num_classes,
    num_steps,
    size_hidden_state,
    dropout
):

    #batchsize = 1
    inputs = []
    shrinked_features = []

    if('first_glance' in features_list):
        input_first_glance = tf.keras.Input(shape=(
            feature2size['first_glance'],), dtype='float32', name='input_first_glance')
        shrink_first_glance = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            feature2size['first_glance'],), activation='relu', use_bias=True, name='shrink_first_glance')
        edges_relu_first_glance = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            size_hidden_state,), activation='relu', use_bias=True, name='edges_relu_first_glance')
        edges_tanh_first_glance = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            size_hidden_state,), activation='tanh', use_bias=True, name='edges_tanh_first_glance')
        inputs.append(input_first_glance)
        shrinked_features.append(shrink_first_glance(input_first_glance))

    if('bodies_age' in features_list):
        input_bodies_age = tf.keras.Input(
            shape=(feature2size['bodies_age'],), dtype='float32', name='input_bodies_age')
        shrink_bodies_age = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            feature2size['bodies_age'],), activation='relu', use_bias=True, name='shrink_bodies_age')
        edges_relu_bodies_age = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            size_hidden_state,), activation='relu', use_bias=True, name='edges_relu_bodies_age')
        edges_tanh_bodies_age = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            size_hidden_state,), activation='tanh', use_bias=True, name='edges_tanh_bodies_age')
        inputs.append(input_bodies_age)
        shrinked_features.append(shrink_bodies_age(input_bodies_age))

    if('bodies_clothing' in features_list):
        input_bodies_clothing = tf.keras.Input(shape=(
            feature2size['bodies_clothing'],), dtype='float32', name='input_bodies_clothing')
        shrink_bodies_clothing = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            feature2size['bodies_clothing'],), activation='relu', use_bias=True, name='shrink_bodies_clothing')
        edges_relu_bodies_clothing = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            size_hidden_state,), activation='relu', use_bias=True, name='edges_relu_bodies_clothing')
        edges_tanh_bodies_clothing = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            size_hidden_state,), activation='tanh', use_bias=True, name='edges_tanh_bodies_clothing')
        inputs.append(input_bodies_clothing)
        shrinked_features.append(shrink_bodies_clothing(input_bodies_clothing))

    if('bodies_gender' in features_list):
        input_bodies_gender = tf.keras.Input(shape=(
            feature2size['bodies_gender'],), dtype='float32', name='input_bodies_gender')
        shrink_bodies_gender = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            feature2size['bodies_gender'],), activation='relu', use_bias=True, name='shrink_bodies_gender')
        edges_relu_bodies_gender = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            size_hidden_state,), activation='relu', use_bias=True, name='edges_relu_bodies_gender')
        edges_tanh_bodies_gender = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            size_hidden_state,), activation='tanh', use_bias=True, name='edges_tanh_bodies_gender')
        inputs.append(input_bodies_gender)
        shrinked_features.append(shrink_bodies_gender(input_bodies_gender))

    if('bodies_activity' in features_list):
        input_bodies_activity = tf.keras.Input(shape=(
            feature2size['bodies_activity'],), dtype='float32', name='input_bodies_activity')
        shrink_bodies_activity = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            feature2size['bodies_activity'],), activation='relu', use_bias=True, name='shrink_bodies_activity')
        edges_relu_bodies_activity = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            size_hidden_state,), activation='relu', use_bias=True, name='edges_relu_bodies_activity')
        edges_tanh_bodies_activity = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            size_hidden_state,), activation='tanh', use_bias=True, name='edges_tanh_bodies_activity')
        inputs.append(input_bodies_activity)
        shrinked_features.append(shrink_bodies_activity(input_bodies_activity))

    if('context_activity' in features_list):
        input_context_activity = tf.keras.Input(shape=(
            feature2size['context_activity'],), dtype='float32', name='input_context_activity')
        shrink_context_activity = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            feature2size['context_activity'],), activation='relu', use_bias=True, name='shrink_context_activity')
        edges_relu_context_activity = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            size_hidden_state,), activation='relu', use_bias=True, name='edges_relu_context_activity')
        edges_tanh_context_activity = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            size_hidden_state,), activation='tanh', use_bias=True, name='edges_tanh_context_activity')
        inputs.append(input_context_activity)
        shrinked_features.append(
            shrink_context_activity(input_context_activity))

    if('context_emotion' in features_list):
        input_context_emotion = tf.keras.Input(shape=(
            feature2size['context_emotion'],), dtype='float32', name='input_context_emotion')
        shrink_context_emotion = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            feature2size['context_emotion'],), activation='relu', use_bias=True, name='shrink_context_emotion')
        edges_relu_context_emotion = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            size_hidden_state,), activation='relu', use_bias=True, name='edges_relu_context_emotion')
        edges_tanh_context_emotion = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            size_hidden_state,), activation='tanh', use_bias=True, name='edges_tanh_context_emotion')
        inputs.append(input_context_emotion)
        shrinked_features.append(shrink_context_emotion(input_context_emotion))

    if('objects_attention' in features_list):
        input_objects_attention = tf.keras.Input(shape=(
            feature2size['objects_attention'],), dtype='float32', name='input_objects_attention')
        shrink_objects_attention = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            feature2size['objects_attention'],), activation='relu', use_bias=True, name='shrink_objects_attention')
        edges_relu_objects_attention = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            size_hidden_state,), activation='relu', use_bias=True, name='edges_relu_objects_attention')
        edges_tanh_objects_attention = tf.keras.layers.Dense(size_hidden_state, input_shape=(
            size_hidden_state,), activation='tanh', use_bias=True, name='edges_tanh_objects_attention')
        inputs.append(input_objects_attention)
        shrinked_features.append(
            shrink_objects_attention(input_objects_attention))

    input_adjacency_matrix = tf.keras.Input(
        shape=(None,), dtype='float32', name='input_adjacency_matrix')
    inputs.append(input_adjacency_matrix)

    #adjacency_matrix = tf.expand_dims(adjacency_matrix , axis=0)
    #adjacency_matrix = tf.repeat(adjacency_matrix, batchsize, axis=0)

    edges_dropout = tf.keras.layers.Dropout(dropout, name='edges_dropout')

    concat_hidden_state = tf.keras.layers.Concatenate(
        axis=0, name='concat_hidden_state')

    propagation_model = tf.keras.layers.GRUCell(
        size_hidden_state, implementation=1, use_bias=True, recurrent_dropout=dropout, name='propagation_model')

    classifier = tf.keras.layers.Dense(size_hidden_state, input_shape=(
        size_hidden_state,), activation='relu', use_bias=True, name='classifier')

    output = tf.keras.layers.Dense(num_classes, input_shape=(
        size_hidden_state,), use_bias=True, name='logits')

    output_dropout = tf.keras.layers.Dropout(
        float(dropout/2.), name='output_dropout')

    hidden_state = concat_hidden_state(shrinked_features)

    for step in range(num_steps):

        messages = []
        tail = 0

        if('first_glance' in features_list):
            head = tail
            tail = tail + 1
            message_first_glance = edges_dropout(edges_tanh_bodies_age(
                edges_relu_bodies_age(hidden_state[head:tail])))
            messages.append(message_first_glance)

        if('bodies_age' in features_list):
            head = tail
            tail = tail + 2
            message_bodies_age = edges_dropout(edges_tanh_bodies_age(
                edges_relu_bodies_age(hidden_state[head:tail])))
            messages.append(message_bodies_age)

        if('bodies_clothing' in features_list):
            head = tail
            tail = tail + 2
            message_bodies_clothing = edges_dropout(edges_tanh_bodies_clothing(
                edges_relu_bodies_clothing(hidden_state[head:tail])))
            messages.append(message_bodies_clothing)

        if('bodies_gender' in features_list):
            head = tail
            tail = tail + 2
            message_bodies_gender = edges_dropout(edges_tanh_bodies_gender(
                edges_relu_bodies_gender(hidden_state[head:tail])))
            messages.append(message_bodies_gender)

        if('bodies_activity' in features_list):
            head = tail
            tail = tail + 2
            message_bodies_activity = edges_dropout(edges_tanh_bodies_activity(
                edges_relu_bodies_activity(hidden_state[head:tail])))
            messages.append(message_bodies_activity)

        if('context_activity' in features_list):
            head = tail
            tail = tail + 1
            message_context_activity = edges_dropout(edges_tanh_context_activity(
                edges_relu_context_activity(hidden_state[head:tail])))
            messages.append(message_context_activity)

        if('context_emotion' in features_list):
            head = tail
            tail = tail + 1
            message_context_emotion = edges_dropout(edges_tanh_context_emotion(
                edges_relu_context_emotion(hidden_state[head:tail])))
            messages.append(message_context_emotion)

        if('objects_attention' in features_list):
            head = tail
            message_objects_attention = edges_dropout(edges_tanh_objects_attention(
                edges_relu_objects_attention(hidden_state[head:])))
            messages.append(message_objects_attention)

        message_combined = concat_hidden_state(messages)
        message_aggregated = tf.matmul(
            input_adjacency_matrix, message_combined)
        #message_aggregated = tf.keras.layers.Dot(axes=(2,1))([input_adjacency_matrix, message_combined])

        hidden_state = propagation_model(message_aggregated, [hidden_state])[0]

    logits = output(output_dropout(classifier(hidden_state)))

    model = tf.keras.Model(inputs=inputs, outputs=logits, name='AGN')
    model.summary(line_length=160)

    return model

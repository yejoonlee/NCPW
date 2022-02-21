import json

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing

from kafka import KafkaConsumer
from kafka import KafkaProducer

from reference_var import reference

reference = reference()
sequence, model_path = reference.get_ref_class()

scaler = preprocessing.MinMaxScaler()
model = tf.keras.models.load_model(model_path)

topicName = 'NCPW-class'
brokers = ["localhost:9091", "localhost:9092", "localhost:9093"]
consumer = KafkaConsumer("NCPW-total", bootstrap_servers=brokers)
producer = KafkaProducer(bootstrap_servers = brokers)

logs = dict()
for message in consumer:
  id = json.loads(message.value.decode('utf-8'))['id']
  states = json.loads(json.loads(message.value.decode('utf-8'))['states'])
  df_states = pd.DataFrame(states)
  scaled_states = scaler.fit_transform(df_states)

  presdict_input = []
  for index in range(len(scaled_states) - sequence):
    presdict_input.append(scaled_states[index: index + sequence])
  presdict_input = np.array(presdict_input)

  class_results = model.predict(presdict_input)

  logs["id"] = id
  logs["class_results"] = str(class_results.tolist())
  logs["class_result_avg"] = str([class_results[:,0].mean(), class_results[:,1].mean(), class_results[:,2].mean()])
  # msg = bytes(str(logs), 'utf-8')
  # print(logs)

  producer.send(topicName, json.dumps(logs).encode("utf-8"))
  producer.flush()
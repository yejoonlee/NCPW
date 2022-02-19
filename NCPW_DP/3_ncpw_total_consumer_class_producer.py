import json
from collections import defaultdict

import tensorflow as tf
import pandas as pd
from sklearn import preprocessing

from kafka import KafkaConsumer
from kafka import KafkaProducer

scaler = preprocessing.MinMaxScaler()
model = tf.keras.models.load_model('/Users/yeznable/Documents/GitHub/NCPW/NCPW_DL/classify_log/GRU_model')
sequence = 4

brokers = ["localhost:9091", "localhost:9092", "localhost:9093"]
consumer = KafkaConsumer("NCPW-total", bootstrap_servers=brokers)
producer = KafkaProducer(bootstrap_servers = brokers)

logs = defaultdict(list)
for message in consumer:
  id = json.loads(message.value.decode('utf-8'))['id']
  states = json.loads(json.loads(message.value.decode('utf-8'))['states'])
  df_states = pd.DataFrame(states)
  scaled_states = scaler.fit_transform(df_states)

  presdict_input = []
  for index in range(len(scaled_states) - sequence):
    presdict_input.append(scaled_states[index: index + sequence])

  class_result = model.predict(presdict_input)

  logs[id].append(class_result)
  print(logs)
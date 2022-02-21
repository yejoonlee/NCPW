import json
from collections import defaultdict
import numpy as np

from kafka import KafkaConsumer
from kafka import KafkaProducer

from reference_var import reference

reference = reference()
update_rate = reference.get_ref_update()

topicName = 'NCPW-class-update'
brokers = ["localhost:9091", "localhost:9092", "localhost:9093"]
consumer_class = KafkaConsumer("NCPW-class", bootstrap_servers=brokers)
consumer_update = KafkaConsumer("NCPW-class-update", bootstrap_servers=brokers)
producer = KafkaProducer(bootstrap_servers = brokers)

class_before_dict = defaultdict(list)
for message in consumer_class:
  class_before = class_before_dict[id]

  id = json.loads(message.value.decode('utf-8'))['id']
  avg = json.loads(json.loads(message.value.decode('utf-8'))['class_result_avg'])
  np_avg = np.array(avg)

  if len(class_before_dict[id]) < 3 :
    class_before_dict[id] = np_avg
    class_updated_list = np_avg
  else:
    class_updated_list = class_before_dict[id]*update_rate + np_avg*update_rate
    class_before_dict[id] = class_updated_list

  logs = dict()
  logs["id"] = id
  try:
    logs["class_before"] = str(class_before.tolist())
  except:
    logs["class_before"] = str(class_before)
  logs["class_updated"] = str(class_updated_list.tolist())

  producer.send(topicName, json.dumps(logs).encode("utf-8"))
  producer.flush()

import json

from kafka import KafkaConsumer
from kafka import KafkaProducer

brokers = ["localhost:9091", "localhost:9092", "localhost:9093"]
consumer = KafkaConsumer("NCPW-total", bootstrap_servers=brokers)
producer = KafkaProducer(bootstrap_servers = brokers)

for message in consumer:
  wl = max([json.loads(json.loads(message.value.decode('utf-8'))['states'])[-1][3], 0])
  if wl:
    producer.send("NCPW-win", message.value)
    producer.flush()
  else:
    producer.send("NCPW-lose", message.value)
    producer.flush()
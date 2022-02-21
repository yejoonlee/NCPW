import json
import time

from kafka import KafkaProducer
from AGENT_helper import *
from reference_var import reference

reference = reference()
id = '[10_10_20_20]'
agent_path = reference.get_agent_path(id)

brokers = ["localhost:9091", "localhost:9092", "localhost:9093"]
topicName = "NCPW-total"

producer = KafkaProducer(bootstrap_servers = brokers)

agent_load = load_agent(agent_path)

while True:
    playLog = get_single_play_log(agent_load)
    playLog['id']=id

    # print(playLog)

    producer.send(topicName, json.dumps(playLog).encode("utf-8"))
    producer.flush()
    time.sleep(10)
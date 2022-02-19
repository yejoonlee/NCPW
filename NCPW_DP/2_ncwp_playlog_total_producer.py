import json
import time

from kafka import KafkaProducer
from AGENT_helper import *

brokers = ["localhost:9091", "localhost:9092", "localhost:9093"]
topicName = "NCPW-total"

producer = KafkaProducer(bootstrap_servers = brokers)

id = '[10_10_20_20]'
agent_load = load_agent(f'/Users/yeznable/Documents/GitHub/NCPW/agents/DQN_agent_{id}.ptb')

while True:
    playLog = get_single_play_log(agent_load)
    playLog['id']=id
    # print(playLog)

    producer.send(topicName, json.dumps(playLog).encode("utf-8"))
    producer.flush()
    time.sleep(10)
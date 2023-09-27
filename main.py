from confluent_kafka import Consumer, Producer, KafkaError, KafkaException
import requests
import json
from config import settings
from logger import get_file_logger


logger = get_file_logger(__name__, 'logs')

consumer_conf = {
    'bootstrap.servers': settings.KAFKA_BOOTSTRAP_SERVERS,
    'group.id': settings.KAFKA_EMOTION_CONSUMER_GROUP,
    'auto.offset.reset': settings.KAFKA_AUTO_OFFSET_RESET,
}

producer_conf = {
    'bootstrap.servers': settings.KAFKA_BOOTSTRAP_SERVERS,
}

consumer = Consumer(consumer_conf)
producer = Producer(producer_conf)


def predict(text :str):
    body = {'inputs': [text]}
    request = requests.post(settings.EMOTION_MODEL_URL, json=body)
    request_json = request.json()
    request_json['text'] = text
    return request_json

def value_serializer(value):
    return json.dumps(value).encode('utf-8') 

def value_deserializer(value):
    return json.loads(value.decode('utf-8'))

def process_msg(msg):
    data = value_deserializer(msg.value())
    output = predict(data['text'])
    if not output['outputs']:
        return
    return value_serializer(output)

running = True

def consume_produce_loop(consumer, consume_topics, produce_topic):
    try:
        consumer.subscribe(consume_topics)

        while running:
            msg = consumer.poll(timeout=1.0)
            if msg is None: 
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.error('%% %s [%d] reached end at offset %d\n' %
                                     (msg.topic(), msg.partition(), msg.offset()))
                elif msg.error():
                    raise KafkaException(msg.error())
            else:
                value = process_msg(msg)
                if not value:
                    continue
                producer.produce(produce_topic, value)
    finally:
        consumer.close()
        producer.flush()

def shutdown():
    running = False

if __name__ == '__main__':
    consume_produce_loop(consumer, [settings.KAFKA_CLEANED_TEXT_TOPIC], settings.KAFKA_EMOTION_TOPIC)

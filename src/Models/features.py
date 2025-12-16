import redis
import os
from config import settings

redis_client = redis.from_url(settings.ADMIN_SECRET)

class RealTimeFeatureEngine:
    def __init__(self, tenant_id):
        self.tenant_id = tenant_id

    def calculate_velocity(self, key_field: str, value: str, window_seconds=3600):
        redis_key = f"velocity:{self.tenant_id}:{key_field}:{value}"
        
        current_count = redis_client.incr(redis_key)
        
        if current_count == 1:
            redis_client.expire(redis_key, window_seconds)
            
        return current_count

    def enrich_transaction(self, transaction: dict):
        
        features = transaction.copy()
        if "card_hash" in transaction:
            features["card_count_1h"] = self.calculate_velocity("card_hash", transaction["card_hash"]) 
        if "ip_address" in transaction:
            features["ip_count_1h"] = self.calculate_velocity("ip", transaction["ip_address"])

        return features
class QwenAPI:
    """Interface for your fine-tuned Qwen2.5 3B model"""
    
    def __init__(self, base_url: str = "http://localhost:8000", model_name: str = "qwen2.5-3b"):
        self.base_url = base_url
        self.model_name = model_name
        raise NotImplementedError("Should Implement the model connection")
    
    # def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
    #     """Generate response using your Qwen model"""
    #     try:
    #         # Adjust this based on your model's API format
    #         response = requests.post(
    #             f"{self.base_url}/v1/completions",
    #             json={
    #                 "model": self.model_name,
    #                 "prompt": prompt,
    #                 "max_tokens": max_tokens,
    #                 "temperature": 0.7,
    #                 "top_p": 0.9
    #             },
    #             timeout=30
    #         )
            
    #         if response.status_code == 200:
    #             return response.json()["choices"][0]["text"].strip()
    #         else:
    #             logger.error(f"API error: {response.status_code}")
    #             return "I apologize, but I'm having trouble generating a response right now."
                
    #     except Exception as e:
    #         logger.error(f"Error calling Qwen API: {e}")
    #         return "I apologize, but I'm having trouble connecting to the model right now."

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TransformerModel:
    def __init__(self, device="cuda"):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.to(device)
        self.model.eval()
        self.device = device

    def generate_text(self, inputs):
        """
        Args:
            inputs (torch.Tensor): The input features of shape (B, F)
        Returns:
            generated_text (list of str): The generated text for each input
        """
        generated_text = []
        with torch.no_grad():
            inputs = inputs.to(self.device)
            for input in inputs:
                input = input.unsqueeze(0)
                generated = self.model.generate(input, max_length=150, temperature=0.7, num_return_sequences=1)
                text = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated]
                generated_text.append(text[0])
        return generated_text

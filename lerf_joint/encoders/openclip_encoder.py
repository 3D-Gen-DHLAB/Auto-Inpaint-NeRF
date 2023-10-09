from dataclasses import dataclass, field
from typing import Tuple, Type
import math

import torch
import torchvision

try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"

from lerf.encoders.image_encoder import (BaseImageEncoder,
                                         BaseImageEncoderConfig)

from nerfstudio.viewer.server.viewer_elements import ViewerText


@dataclass
class OpenCLIPNetworkConfig(BaseImageEncoderConfig):
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    """
    room_positives: Tuple[str] = ("wall", "floor", "doorframe", "door", "ceiling", "support beam", "support column", "building", "window", "windowsill")
    room_negatives: Tuple[str] = ("desk", "chair", "couch", "pictureframe", "painting", "poster", "bench", "furniture",
        "instrument", "bookshelf", "curtain", "curtains", "bed", "stool", "computer", "laptop", "speakers", "computer equipment", "television",
        "bedside table", "dinner table", "computer screen", "trash", "cardboard box", "phone", "object", "stuff", "things")
        room_positives: Tuple[str] = ("wall", "floor", "doorframe", "door", "ceiling", "support beam", "support column", "building", "window", "windowsill")
    room_negatives: Tuple[str] = ("desk", "table", "furniture",  "trash", "cardboard box", "object")
    """
    room_positives: Tuple[str] = ("wall", "floor",  "door", "doorframe", "ceiling", "window")
    #room_positives: Tuple[str] = ("object", "stuff", "things", "texture")
    room_negatives: Tuple[str] = ("object", "stuff", "things", "texture", "desk", "table", "furniture",  "trash", "chair", "box", "cardboard box",  
         "bookshelf", "lamp", "light fixture", "cable", "poster", "artwork", "computer", "television", "couch", "shelf")
   

class OpenCLIPNetwork(BaseImageEncoder):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positive_input = ViewerText("LERF Positives", "", cb_hook=self.gui_cb)

        self.positives = self.positive_input.value.split(";")
        
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        self.positives_room = self.config.room_positives
        self.negatives_room = self.config.room_negatives

        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives_room]).to("cuda")
            self.pos_embeds_room = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives_room]).to("cuda")
            self.neg_embeds_room = model.encode_text(tok_phrases)
        self.pos_embeds_room /= self.pos_embeds_room.norm(dim=-1, keepdim=True)
        self.neg_embeds_room /= self.neg_embeds_room.norm(dim=-1, keepdim=True)

        
        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        print(self.positives)
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
            :, 0, :
        ]

    def get_room_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        
        phrases_embeds = torch.cat([self.pos_embeds_room, self.neg_embeds_room], dim=0)

        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        
        output = torch.softmax(10 *output, dim=-1)
        
        positive_vals = output[..., positive_id : positive_id + len(self.positives_room)]  # rays x 1

        negative_vals = output[..., len(self.positives_room) :]  # rays x N_phrase
        
        pos_len = len(self.positives_room)
        neg_len = len(self.negatives_room)
        
        if neg_len > pos_len:
            
            repeat_num = math.ceil(neg_len/pos_len)
            repeated_pos = positive_vals.repeat(1, repeat_num)[..., :negative_vals.shape[1]]  # rays x N_phrase

            sims = torch.stack((repeated_pos, negative_vals), dim=-1)  
            pos_len = repeat_num
        elif neg_len < pos_len:

            repeat_num = math.ceil(pos_len/neg_len)
            repeated_neg = negative_vals.repeat(1, repeat_num)[..., :positive_vals.shape[1]]
            
            sims = torch.stack((positive_vals, repeated_neg), dim=-1)  
            neg_len = repeat_num
        elif neg_len == pos_len:
            sims = torch.stack((positive_vals, negative_vals), dim=-1)  
        #softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        """
        
        !!!
        print()
        print(softmax.shape)
        print(softmax[0, :, 0])
        print(softmax[0, :, 1])
        best_ids = torch.max(softmax, dim=1).values
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        out = torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives_room), 2))[
            :, 0, :
        ]
        !!!
        

        """
        best_ids = torch.max(sims, dim=1).values
        
        return best_ids

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)

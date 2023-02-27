import torch
from prompters import *


class TaskVector:
    def __init__(self, args):
        """Initializes the task vector from a finetuned prompter checkpoint.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        self.prompter = self.get_prompter(args, identity=True)

        if args.vector is not None:
            self.vector = args.vector
        else:
            assert args.finetuned_prompter_checkpoint is not None
            with torch.no_grad():
                finetuned_state_dict = torch.load(args.finetuned_prompter_checkpoint).state_dict()
                self.vector = {}
                for key in finetuned_state_dict:
                    if finetuned_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key]

    def get_prompter(self, args, identity=False):
        if args.prompter_type == "padding":
            return padding(args, identity)
        elif args.prompter_type == "fixed_patch":
            return fixed_patch(args, identity)
        elif args.prompter_type == "random_patch":
            return random_patch(args, identity)
        else:
            return NotImplementedError

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            new_state_dict = {}
            pretrained_state_dict = self.prompter.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(
                        f"Warning: key {key} is present in the pretrained state dict but not in the task vector"
                    )
                    continue
                new_state_dict[key] = scaling_coef * self.vector[key]
        self.prompter.load_state_dict(new_state_dict, strict=False)
        return self.prompter

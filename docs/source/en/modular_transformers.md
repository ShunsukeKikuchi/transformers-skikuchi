# Modular transformers

`transformers` is an opinionated framework; our philosophy is defined in the following [conceptual guide](./philosophy).

The core of that philosophy is exemplified by the [single model, single file](https://huggingface.co/blog/transformers-design-philosophy)
aspect of the library. This component's downside is that it limits the inheritance and importability of components from
files to others in the toolkit.

As a result, model components tend to be repeated across many files. There are as many attention layers defined
in `transformers` as there are models, and a significant number of those are identical to each other. 
The unfortunate consequence is that independent implementations tend to diverge as fixes and changes get applied
to specific parts of the code.

In order to balance this issue, we introduced the concept of "copies" across the library. By adding a comment indicating
that code is a copy of another, we can enforce through CI and local commands that copies do not diverge. However,
while the complexity is low, this is often quite tedious to do.

And, finally, this contributes to adding a significant overhead to contributing models which we would like to remove.
This approach often requires model contributions to add modeling code (~1k lines), processor (~500 lines), tests, docs,
etc. Model contribution PRs rarely add less than 3-5k lines of code, with much of this code being boilerplate.

This raises the bar for contributions, and with Modular Transformers, we're aiming to lower the bar to a much more
acceptable point.

If you plan to add a model to `transformers` make sure you read [How to add a model to 🤗 Transformers?](https://huggingface.co/docs/transformers/add_new_model).
For any kind of contributions, see [CONTRIBUTING.md](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md).

## What is it?

Modular Transformers introduces the concept of a "modular" file to a model folder. This modular file accepts code
that isn't typically accepted in modeling/processing files, as it allows importing from neighbouring models as well
as inheritance from classes to others.

This modular file defines models, processors, and the configuration class that would otherwise be defined in their
respective modules.

Finally, this feature introduces a new `linter` which will "unravel" the modular file into the "single model, single 
file" directory structure. These files will get auto-generated every time the script is run; reducing the required
contributions to the modular file, and therefore only to the changes between the contributed model and others.

Model users will end up importing and using the single-file interface, so no change is expected here. Doing this, we
hope to combine the best of both worlds: enabling simple contributions while sticking to our philosophy.

This is therefore a replacement for the `# Copied from` markers, and previously contributed models can be expected to
be moved to the new Modular Transformers format in the coming months.

### Details 

To generate a single file from the modular file, run the following command.

```bash
python utils/modular_model_converter.py --files-to-parse src/transformers/models/<your_model>/modular_<your_model>.py
```

The "linter", which unravels the inheritance and creates all single-files from the modular file, will flatten the 
inheritance while trying to be invisible to Python users. At this time, the linter flattens a **single** level of
inheritance.

For example:
- If a configuration class inherits from another and adds/deletes an argument, the generated file will either directly 
  reference it (in case of addition) or completely remove it (in case of deletion).
- If a class inherits from another, for example: `class GemmaModel(LlamaModel):`, dependencies are automatically 
  inferred. All submodules will be automatically added from the superclass.
- If you define new functions in the `modular` and use them inside classes, the linter will automatically infer the 

You should be able to write everything (the tokenizer, the image processor, the model, the config) in this `modular` 
file, and the corresponding files will be created for you. 

### Enforcement

Run the command below to ensure the generated content matches `modular_<your_model>.py`

```bash
python utils/check_modular_conversion.py --files src/transformers/models/<your_model>/modular_<your_model>.py
```

### Examples

Here is a quick example with BERT and RoBERTa. The two models are intimately related: their modeling implementation 
differs solely by a change in the embedding layer.

Instead of redefining the model entirely, here is what the `modular_roberta.py` file looks like for the modeling &
configuration classes (for the sake of the example, the tokenizer is ignored at this time as very different).

```python
from torch import nn
from ..bert.configuration_bert import BertConfig
from ..bert.modeling_bert import (
    BertModel,
    BertEmbeddings,
    BertForMaskedLM
)

# The RoBERTa config is identical to BERT's config
class RobertaConfig(BertConfig):
  model_type = 'roberta'

# We redefine the embeddings here to highlight the padding ID difference, and we redefine the position embeddings
class RobertaEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config())

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

# The RoBERTa model is identical to the BERT model, except for the embedding layer. 
# We redefine the embeddings above, so here there is no need to do additional work
class RobertaModel(BertModel):
  def __init__(self, config):
    super().__init__(config)
    self.embeddings = RobertaEmbeddings(config)

      
# The heads now only need to redefine the model inside to the correct `RobertaModel`
class RobertaForMaskedLM(BertForMaskedLM):
  def __init__(self, config):
    super().__init__(config)
    self.model = RobertaModel(config)
```

## What it is not

It is not a replacement for the modeling code (yet?), and if your model is not based on anything else that ever existed, then you can add a `modeling` file as usual. Similarly, if you cannot easily inherit your `configuration` (or `tokenization` or `processing`) file from another model's similar file, you can add that filetype directly (even though defining it in the modular file would work, it would clutter it).


## Real world example breakdown

As explained, modular allows you to use regular Python inheritance from any other model's code in the library, in order to define your own. For this reason, it will work better/be easier if you first browse the library a bit to find models close to yours, in order to inherit from them. For example, are you using a sliding window in the `Attention` class? Then start by checking well known models doing so, e.g. Mistral, or Qwen2! Are you using interleaved `RotaryEmbedding` modules? Check out Cohere and Cohere2 models! Otherwise a very strong starting point is to check out Llama. And if you are doing a bit of all of that at once, then you can mix and match!

At Hugging Face, we feel that learning by example is usually (one of) the best way, so we will now go over a typical modular file, and the different features our linter provides (and its limitations)! 🤗 Let's use a real world example with Olmo2 model, which I feel provides a very good illustration of the modular mechanisms. The original file can be found [here](https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmo2/modular_olmo2.py). For simplicity, we will go over it class by class, and repeat the modular's definition of ech class.

Let's break it down!


### Config class

Here is the `Config` definition in modular:

```py
from ..olmo.configuration_olmo import OlmoConfig

class Olmo2Config(OlmoConfig):
    r"""
    This is the configuration class to store the configuration of a [`Olmo2Model`].
    """

    def __init__(
        self,
        vocab_size=50304,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=None,
        eos_token_id=50279,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            **kwargs,
        )

        self.rms_norm_eps = rms_norm_eps
        del self.clip_qkv
```

Here, we correctly identified that the `Config` in Olmo2 is similar to Olmo's, up to a few details:
1. The default value of most arguments has changed
2. we have a new argument, `rms_norm_eps`
3. the argument `clip_qkv` is not used anymore

To solve points 1. and 2., simply overwriting the `__init__` function with the new default arguments and adding the new one is enough, as you would expect when you want to overwrite a method in Python! Of course you also need to assign the new attribute `rms_norm_eps` to `self` in the `__init__`'s body.  
For point 3., we use the special syntax `del self.clip_qkv`, which, has you can expect, removed the assignment of this attribute in the unravelled code (after the conversion with the linter).  

Now, there is a subtility here: as you can see, we used `super().__init__(...)`. Usually, in Python, it is simply used to call the parent's `__init__`. In modular terms, however, it has a _slightly_ different meaning. When we find a call such as `super().my_function(...)` in the modular file, the linter will take the body of the `my_function` function in the parent, and unravel it where the call to `super().my_function(...)` occured. Then, the `del self.clip_qkv` statement will remove the reference to `self.clip_qkv` from the unravelled body. Thus `del self.xxx` can only work in pair with `super().my_function(...)`, and should always be placed after it (but you can add whatever you want _before_ calling `super()`, and it will be placed, as you can expect, before the parent's body).

### Norm class

Here is the `Norm` class:

```py
from ..llama.modeling_llama import LlamaRMSNorm

class Olmo2RMSNorm(LlamaRMSNorm):
    pass
```

What to say here, it is pretty explicit isn't it? We do not modify anything from the `LlamaRMSNorm` definition. Thus the linter will unravel exactly the content of the parent (`LlamaRMSNorm`). Only change will be that every reference to "llama" on the docstrings, type hints, and comments (basically everywhere) will be changed to references to "olmo2" for consistency!

### Attention class

Here is the `Attention` class:

```py
from ..llama.modeling_llama import eager_attention_forward
from ..olmo.modeling_olmo import OlmoAttention, apply_rotary_pos_emb


# Olmo2 attention is identical to OLMo attention except:
# - Norm is applied to attention queries and keys.
# - No qkv clipping.
class Olmo2Attention(OlmoAttention):
    def __init__(self, config: Olmo2Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx=layer_idx)
        self.q_norm = Olmo2RMSNorm(config.num_attention_heads * self.head_dim, config.rms_norm_eps)
        self.k_norm = Olmo2RMSNorm(config.num_key_value_heads * self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
```

Now, what's happening here? In the `__init__`, we call `super().__init__(...)`, thus copying the parent's definition, then add 2 new layers of the `Olmo2RMSNorm` we just added previously. Indeed, those were not present in the original `Olmo` (v1) model. So, now, we also have to overwrite the `forward` method to use these 2 new layers right? Indeed, if you check carefully, the definition of `forward` is identical to `Olmo`'s, but we added a pass with the norm layers just before projecting with `q_proj` and `k_proj`. However, to help us, we directly imported the functions `eager_attention_forward` from llama, and `apply_rotary_pos_emb` from olmo. The linter will then automatically add these imported functions in the final `modeling_olmo2.py` file, by copying their definitions from the source (imported) files. And it will even add the `rotate_half` and `repeat_kv` functions (which are used inside `apply_rotary_pos_embed` and `eager_attention_forward` respectively) by figuring out the dependency automatically. Neat, right?  
Note that we had to redefine this class, because we did not find any model defining the `Attention` layer with the added `RMSNorm` layer anywhere else in the library! Otherwise, we would have simply inherited from this model instead as we did for the `RMSNorm`!

## The DecoderLayer class

Here is the `DecoderLayer` class:

```py
from ..olmo.modeling_olmo import OlmoDecoderLayer

# The OLMo2 layers are identical to those of the OLMo model except:
# - RMSNorm is used instead of standard layer norm.
# - Norm is applied after attention/feedforward rather than before.
class Olmo2DecoderLayer(OlmoDecoderLayer):
    def __init__(self, config: Olmo2Config, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        self.post_attention_layernorm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Olmo2Attention(config=config, layer_idx=layer_idx)
        del self.input_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs
```

At this point, you should start to pick up what is happening for this class. We switched the type of norm in the `__init__` by overwriting `self.post_attention_layernorm` after the call to `super().__init__(...)`, thus going from a `LayerNorm` in the parent class, to our `RMSNorm` in this class. Then we simply deleted the `self.input_layernorm` attribute, and replaced it by `self.post_feedforward_layernorm`, because the name was not making sense anymore as we apply it after in `Olmo2` instead of before in `Olmo`. For this reason, we also need to overwrite the `forward` method, to reflect the logic change.

Note however that if we had only switched `self.post_attention_layernorm` and `self.input_layernorm` from `LayerNorm`s to `RMSNorm`s (without the name and logic change of `elf.input_layernorm`), we would not have had to redefine the `forward` method!

## The Model class

```py
from ..olmo.modeling_olmo import OlmoModel

# The OLMo2 model is identical to the OLMo model, except RMSNorm is used instead of
# standard layer norm for the output norm.
class Olmo2Model(OlmoModel):
    def __init__(self, config: Olmo2Config):
        super().__init__(config)
        self.norm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [Olmo2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
```

Here, this is exactly what I was pointing out before: we simply change the _type_ of the `self.norm` attribute (going from `LayerNorn` in `Olmo` to `RMSNorm` in `Olmo2`). Since this change does not reflect the logic of the `forward` method (the name of the layer and where it is used is identical to the parent's), then we do not even need to overwrite it! It will be unravelled automatically! Note that we redefined `self.layers` for the sake of being explicit, but this is not even strictly required here as the definition is similar to what is found in `Olmo` (v1).

## Finally... The ForCausalLM class

Finally, here is the definition of the `ForCausalLM`:

```py
from ..olmo.modeling_olmo import OlmoForCausalLM

class Olmo2ForCausalLM(OlmoForCausalLM):
    pass
```

As for the `RMSNorm`, it is exactly similar to the parent's in logic, so we do not have anything to do, the linter will all figure it out by itself. Almost disappointing, no?

## But... What about the MLP, RotaryEmbedding and PreTrainedModel classes?

Indeed, if you inspect the file [`modeling_olmo2.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmo2/modeling_olmo2.py) which is created by running the linter on `modular_olmo2.py`, you will notice that it also creates `Olmo2MLP`, `Olmo2RotaryEmbedding`, and `Olmo2PreTrainedModel` classes, that we did not define explicitly in `modular_olmo2.py`.  

Well, it is one of the main feature of our modular linter. Similarly to how some functions were added automatically with the `Attention` class (without directly importing them), classes that are a dependency of one of the class inherited class and which are not explicitly defined in the modular file, will be added automatically as part of the dependeny tracing. For example, in `OlmoDecoderLayer`, there is an attribute defined as `self.mlp = OlmoMLP(config)`. Because we never explicitly redefined a class named `Olmo2MLP` in `modular_olmo2.py`, the linter automatically created a class `Olmo2MLP`, similar to `OlmoMLP`. This is exactly the same as if we had done:

```py
from ..olmo.modeling_olmo import OlmoMLP

class Olmo2MLP(OlmoMLP):
    pass
```

but we did not even bother, because we _know_ this class is supposed to be exactly similar, and we never needed it anywhere else in the `modular_olmo2.py` file. In contrast, the class `Olmo2RMSNorm` was needed to (re)define the norms both in the `Attention` and `DecoderLayer` classes. The same logic is true for the `Olmo2PreTrainedModel` and `Olmo2RotaryEmbedding` classes.

Note however that if not redefined, classes will be copied from the file in which an inherited module uses them first. So if you wanted e.g. `Olmo2MLP` to inherit from, say, `MistralMLP` instead of `OlmoMLP` (here it was `OlmoMLP` because it was first implicitly used in `Olmo2DecoderLayer`, which inherited from `OlmoDecoderLayer`), you would need to be explicit and do:

```py
# switch to mistral definition
from ..mistral.modeling_mistral import MistralMLP

class Olmo2MLP(MistralMLP):
    pass
```

## Advanced usage

### Removing attributes and functions
To remove attributes that are not used in your modular model, and that you don't want to see in the unravelled modeling: 

```python
class GemmaModel(LlamaModel):                 |           class GemmaModel(PreTrainedModel):
    def __init__(self, config):               |              def __init__(self, config):
        super().__init__(self, eos_token)     |                 super().__init__(config)
        del self.embed_tokens                 |                 self.padding_idx = config.pad_token_id
                                              |                 self.vocab_size = config.vocab_size
                                              |
                                              |                 self.layers = nn.ModuleList(
                                              |                     [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
                                              |                 )
                                              |                 self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                                              |                 self.rotary_emb = LlamaRotaryEmbedding(config=config)
                                              |                 self.gradient_checkpointing = False
                                              |                 
                                              |                 # Initialize weights and apply final processing
                                              |                 self.post_init()
```
If you check the original `LlamaModel`, it has a `embed_tokens` which was removed here (as you would expect!)

Removing a function is pretty similar, you just need to write it with a `raise ValueError("")` to mimick the behaviour you actually want when you remove a parent function in python.

```python
class GemmaTokenizer(LlamaTokenizer):
    ...

    def get_spm_processor(self):
        raise AttributeError("Not needed for Gemma")

    def unk_token_length(self):
        raise AttributeError("Not needed for Gemma")
```

### Define new functions

If you define a new function in the `modular` file to be used inside a class, say

```python
def my_new_function(*args, **kwargs):
  # Do something here
  pass

class GemmaModel(LlamaModel):
    def forward(*args, **kwargs):
      # Call the function
      example = my_new_function(*args, **kwargs)
      # continue here
```

the `my_new_function` function (and, recursively, any other new functions called in its body) will be automatically copy-pasted 
in the file where it is used.

### Calling `super()`
We recently shipped a few features that allow you to go from:
```python
class GemmaTokenizer(LlamaTokenizer, PretrainedTokenizerFast):         |           class GemmaModel(nn.Module):
    def __init__(self, eos_token="</s>"):                              |             def __init__(self):
        eos_token = AddedToken(eos_token)                              |                eos_token = AddedToken(eos_token)
        PretrainedTokenizerFast.__init__(self, eos_token)              |                super().__init__(eos_token)
```
This is useful want you **don't** want to unravel the call to `super()`, and you want to differentiate which super init call you are doing!

### Special naming
We now also support special cases like
```python
class GemmaVisionModel(CLIPModel):                                 
    pass
```
where the name of your class `GemmaVision` is not the same as the modular `Gemma`. This is super useful for composite models.

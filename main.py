from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Header, Footer, Static, ListView, Label, ListItem, Input, Button
from textual import events, on
from textual.screen import Screen
from textual.message import Message

from pathlib import Path
from os import walk

from safetensors import safe_open
from safetensors.torch import save_file

import torch

class ErrorScreen(Screen):
    BINDINGS = [("escape", "app.pop_screen", "Pop screen")]

    def __init__(self, error_message):
        self.error_message = error_message
        super().__init__()
    
    def compose(self) -> ComposeResult:
        yield Static(self.error_message)
        yield Static("Press ESC to continue [blink]_[/]", id="any-key")

class ModelEntry(Static):
    def __init__(self, models : list[str]):
        super().__init__()
        self.models = models

    def on_mount(self) -> None:
        for model in self.models:
            self.list_view.append(ListItem(Label(model)))

    def compose(self) -> ComposeResult:
        self.list_view = ListView(id='listmodels')
        self.ratio = Input(value='0.5', id='ratio_input')

        yield self.list_view
        yield self.ratio

class BottomButtons(Static):
    class Pressed(Message):
        def __init__(self, button : str) -> None:
            self.button = button
            super().__init__()

    def compose(self) -> ComposeResult:
        yield Button('Add model', id='add_button')
        yield Button('Remove model', id='remove_button')
        yield Button('Merge models', id='merge_button')

    def on_button_pressed(self, event : Button.Pressed):
        self.post_message(BottomButtons.Pressed(event.button.id))

class MergeApp(App):
    CSS_PATH = "styles/mergeapp.tcss"

    def __init__(self):
        super().__init__()

        # first find all the models in the models folder
        models_folder = Path('models')
        self.models = []

        for (dirpath, dirnames, filenames) in walk(models_folder):
            for filename in filenames:
                suffix = Path(filename).suffix

                if suffix == '.safetensors':
                    self.models.append(filename)

        self.model_entries = []
    
    def on_mount(self):
        for i in range(2):
            model_entry = ModelEntry(self.models)
            self.model_entries.append(model_entry)
            self.model_entries_countainer.mount(model_entry)
    
    def compose(self) -> ComposeResult:
        self.model_entries_countainer = Static(id='model_entries')
        yield self.model_entries_countainer
        yield BottomButtons()

    def on_bottom_buttons_pressed(self, message : BottomButtons.Pressed) -> None:
        if message.button == 'add_button':
            model_entry = ModelEntry(self.models)
            self.model_entries.append(model_entry)
            self.model_entries_countainer.mount(model_entry)
        elif message.button == 'remove_button':
            if len(self.model_entries) > 2:
                model_entry = self.model_entries.pop()
                model_entry.remove()
        elif message.button == 'merge_button':
            self.merge_models()
    
    def convert_onetrainer_model_to_diffusers(self, state_dict : dict[str, torch.Tensor]):
        new_state_dict = {}

        # convert the layers outside the transformer
        new_state_dict['proj_out.bias'] = state_dict.pop('final_layer.linear.bias')
        new_state_dict['proj_out.weight'] = state_dict.pop('final_layer.linear.weight')
        new_state_dict['scale_shift_table'] = state_dict.pop('final_layer.scale_shift_table')
        new_state_dict['pos_embed.proj.weight'] = state_dict.pop('pos_embed')

        new_state_dict['adaln_single.linear.bias'] = state_dict.pop('t_block.1.bias')
        new_state_dict['adaln_single.linear.weight'] = state_dict.pop('t_block.1.weight')

        new_state_dict['adaln_single.emb.timestep_embedder.linear_1.bias'] = state_dict.pop('t_embedder.mlp.0.bias')
        new_state_dict['adaln_single.emb.timestep_embedder.linear_1.weight'] = state_dict.pop('t_embedder.mlp.0.weight')
        new_state_dict['adaln_single.emb.timestep_embedder.linear_2.bias'] = state_dict.pop('t_embedder.mlp.2.bias')
        new_state_dict['adaln_single.emb.timestep_embedder.linear_2.weight'] = state_dict.pop('t_embedder.mlp.2.weight')

        new_state_dict['pos_embed.proj.bias'] = state_dict.pop('x_embedder.proj.bias')
        new_state_dict['pos_embed.proj.weight'] = state_dict.pop('x_embedder.proj.weight')

        new_state_dict['caption_projection.linear_1.bias'] = state_dict.pop('y_embedder.y_proj.fc1.bias')
        new_state_dict['caption_projection.linear_1.weight'] = state_dict.pop('y_embedder.y_proj.fc1.weight')
        new_state_dict['caption_projection.linear_2.bias'] = state_dict.pop('y_embedder.y_proj.fc2.bias')
        new_state_dict['caption_projection.linear_2.weight'] = state_dict.pop('y_embedder.y_proj.fc2.weight')

        # detect the transformer depth
        depth = 1
        for key in state_dict.keys():
            if 'blocks.' in key:
                current_depth = int(key.split('.')[1])
                if current_depth > depth - 1:
                    depth = depth + 1
        
        for d in range(depth):
            attn_q, attn_k, attn_v = torch.chunk(state_dict.pop(f'blocks.{d}.attn.qkv.bias'), 3)
            new_state_dict[f'transformer_blocks.{d}.attn1.to_q.bias'] = attn_q
            new_state_dict[f'transformer_blocks.{d}.attn1.to_k.bias'] = attn_k
            new_state_dict[f'transformer_blocks.{d}.attn1.to_v.bias'] = attn_v

            attn_q, attn_k, attn_v = torch.chunk(state_dict.pop(f'blocks.{d}.attn.qkv.weight'), 3)
            new_state_dict[f'transformer_blocks.{d}.attn1.to_q.weight'] = attn_q
            new_state_dict[f'transformer_blocks.{d}.attn1.to_k.weight'] = attn_k
            new_state_dict[f'transformer_blocks.{d}.attn1.to_v.weight'] = attn_v

            new_state_dict[f'transformer_blocks.{d}.attn1.to_out.0.bias'] = \
                state_dict.pop(f'blocks.{d}.attn.proj.bias')
            new_state_dict[f'transformer_blocks.{d}.attn1.to_out.0.weight'] = \
                state_dict.pop(f'blocks.{d}.attn.proj.weight')

            cross_attn_k, cross_attn_v = torch.chunk(state_dict.pop(f'blocks.{d}.cross_attn.kv_linear.bias'), 2)
            cross_attn_q = state_dict.pop(f'blocks.{d}.cross_attn.q_linear.bias')
            new_state_dict[f'transformer_blocks.{d}.attn2.to_k.bias'] = cross_attn_k
            new_state_dict[f'transformer_blocks.{d}.attn2.to_q.bias'] = cross_attn_q
            new_state_dict[f'transformer_blocks.{d}.attn2.to_v.bias'] = cross_attn_v

            cross_attn_k, cross_attn_v = torch.chunk(state_dict.pop(f'blocks.{d}.cross_attn.kv_linear.weight'), 2)
            cross_attn_q = state_dict.pop(f'blocks.{d}.cross_attn.q_linear.weight')
            new_state_dict[f'transformer_blocks.{d}.attn2.to_k.weight'] = cross_attn_k
            new_state_dict[f'transformer_blocks.{d}.attn2.to_q.weight'] = cross_attn_q
            new_state_dict[f'transformer_blocks.{d}.attn2.to_v.weight'] = cross_attn_v

            new_state_dict[f'transformer_blocks.{d}.attn2.to_out.0.bias'] = \
                state_dict.pop(f'blocks.{d}.cross_attn.proj.bias')
            new_state_dict[f'transformer_blocks.{d}.attn2.to_out.0.weight'] = \
                state_dict.pop(f'blocks.{d}.cross_attn.proj.weight')
            
            new_state_dict[f'transformer_blocks.{d}.ff.net.0.proj.bias'] = \
                state_dict.pop(f'blocks.{d}.mlp.fc1.bias')
            new_state_dict[f'transformer_blocks.{d}.ff.net.0.proj.weight'] = \
                state_dict.pop(f'blocks.{d}.mlp.fc1.weight')
            new_state_dict[f'transformer_blocks.{d}.ff.net.2.bias'] = \
                state_dict.pop(f'blocks.{d}.mlp.fc2.bias')
            new_state_dict[f'transformer_blocks.{d}.ff.net.2.weight'] = \
                state_dict.pop(f'blocks.{d}.mlp.fc2.weight')
            
            new_state_dict[f'transformer_blocks.{d}.scale_shift_table'] = \
                state_dict.pop(f'blocks.{d}.scale_shift_table')
        return new_state_dict
    
    def is_onetrainer_model(self, state_dict):
        for key in state_dict.keys():
            if 'final_layer' in key:
                return True
        return False
    
    def merge_models(self):
        # get the ratios
        ratios = []
        sum = 0
        for model_entry in self.model_entries:
            value = float(model_entry.ratio.value)
            sum = sum + value
            ratios.append(value)
        
        # normalize the ratios
        ratios = [ratio / sum for ratio in ratios]

        # now go through the models, load and store the weights
        state_dict = {}

        # load the first model
        first_model_entry = self.model_entries[0]
        index = first_model_entry.list_view.index
        model_filename = self.models[index]

        models_path = Path('models')
            
        first_model_path = models_path.joinpath(model_filename)

        with safe_open(first_model_path, framework='pt', device='cpu') as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key) * ratios[0]
            
            # onetrainer compatibility
            if self.is_onetrainer_model(state_dict):
                state_dict = self.convert_onetrainer_model_to_diffusers(state_dict)
        
        error = ''
        # repeat this with all the other models
        for i in range(1, len(self.model_entries)):
            model_entry = self.model_entries[i]
            index = model_entry.list_view.index

            model_filename = self.models[index]
            model_path = models_path.joinpath(model_filename)

            with safe_open(model_path, framework='pt', device='cpu') as f:
                new_state_dict = {}
                for key in f.keys():
                    new_state_dict[key] = f.get_tensor(key)
                
            # onetrainer compatibility
            if self.is_onetrainer_model(new_state_dict):
                new_state_dict = self.convert_onetrainer_model_to_diffusers(new_state_dict)

            for key in state_dict.keys():
                try:
                    state_dict[key] = state_dict[key] + new_state_dict[key] * ratios[i]
                except Exception as e:
                    error = error + str(e) + '\n'
        
        if error != '':
            self.push_screen(ErrorScreen(error))
        
        # save the model
        save_file(state_dict, 'out.safetensors')


        
        

if __name__ == '__main__':
    app = MergeApp()
    app.run()
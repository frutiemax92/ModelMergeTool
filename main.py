from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Header, Footer, Static, ListView, Label, ListItem, Input, Button
from textual import events, on
from textual.message import Message

from pathlib import Path
from os import walk

from safetensors import safe_open
from safetensors.torch import save_file

import torch

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
            
        # repeat this with all the other models
        for i in range(1, len(self.model_entries)):
            model_entry = self.model_entries[i]
            index = model_entry.list_view.index

            model_filename = self.models[index]
            model_path = models_path.joinpath(model_filename)

            with safe_open(model_path, framework='pt', device='cpu') as f:
                for key in f.keys():
                    try:
                        state_dict[key] = state_dict[key] + f.get_tensor(key) * ratios[i]
                    except:
                        continue
        
        # save the model
        save_file(state_dict, 'out.safetensors')


        
        

if __name__ == '__main__':
    app = MergeApp()
    app.run()
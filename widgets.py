import os
from collections import OrderedDict as od
import ipywidgets as ipw
from copy import deepcopy
import pandas as pd
import traitlets
try:
    from tkinter import Tk, filedialog
    tkinter_available = True
except:
    tkinter_available =False
    
import helper_funcs as helpers
from traceback import format_exc
import numpy as np
import matplotlib.pyplot as plt



### WORKING
class TableEditor(object):        
    _base_layout = ipw.Layout(flex='0 1 auto', width='200px', height='150px')
    _btn_width = "100px"
    def __init__(self, df, save_dir=None, preconfig_file=None, 
                 default_group=None, new_run_names=[], add_to_index_vars=[], 
                 unstack_indices=[], run_level_idx=0, var_level_idx=2, 
                 **plot_settings):
        
        # Stuff for I/O
        if save_dir is None:
            save_dir = os.getcwd()
        
        self.saveas_funs = dict(csv = self.save_csv, 
                                xlsx = self.save_xlsx,
                                png  = self.save_png)
        
        self.plot_funs = dict(heatmap = self.plot_heatmap)
        
        self.save_dir = save_dir
        
        # the dataframe
        self.df = df
        self.df_edit = self.check_shape_init(df)
        self._df_edit_last = self.df_edit
        
        self.extractions = od()
        
        self.var_level_idx = var_level_idx
        self.run_level_idx = run_level_idx
        
        # Predefined settings for things applied to dataframe
        self.new_run_names_init = new_run_names
        self.add_to_index_vars = add_to_index_vars
        self.unstack_indices = unstack_indices
        
        # Display of Dataframe
        self.current_plot = None
        self.heatmap_settings = od(cmap="bwr",
                                   cmap_shifted=True)
        
        # Settings for Variable selector
        self.groups = od()
        self.groups["flagged"] = self.flagged_vars
        if preconfig_file:
            self.groups.update(helpers.load_varconfig_ini(preconfig_file))
        
        if default_group is None:
            default_group = "flagged"
        
        if not default_group in self.groups:
            raise ValueError("No default group with ID {} in file {}".format(default_group, preconfig_file))
            
        self.default_group = default_group
        self.default_selection = self.groups[default_group]
        
        self._buttons_edit_df = []
        # init widgets and actions
        self.init_widgets_renamer()
        self.init_layout_renamer()
        self.init_widgets_varselect()
        self.init_layout_varselect()
        self.init_layout_reshaper()
        self.init_glob_widgets()
        self.init_layout()
        # initiate layout
        
        self.apply_changes_rename()
        self.crop_var_selection()
        self.add_to_index(self.add_to_index_vars)
        self.unstack(self.unstack_indices)
        self.update_ui()
        self.disp_current()
        
        self.heatmap_settings.update(plot_settings)
         
        if not tkinter_available:
            self.btn_saveas.disabled = True
            self.btn_saveas.tooltip = ("Please install tkinter to use this "
                                       "feature. Until then, you can use save "
                                       "button")
        
    @property 
    def default_plot_fun(self):
        return self.plot_funs["heatmap"]
    
    @property
    def column_names(self):
        return list(self.df_edit.columns)
    
    @property
    def data_column_names(self):
        df = self.df_edit
        if isinstance(df.columns, pd.MultiIndex):
            return list(df.columns.levels[0])
        return list(df.columns)
    
    @property
    def index_level_names(self):
        return self.df_edit.index.names
    
    @property
    def index_level_col_names(self):
        return self.df_edit.columns.names[1:]
    
    @property
    def run_names(self):
        #return sorted(self.df.index.get_level_values(self.level).unique().values)
        return self.df_edit.index.get_level_values(self.run_level_idx).unique().values
    
    @property
    def flagged_vars(self):
        lvl = self.var_level_idx
        return list(self.df[self.df.Flag.astype(bool)].index.get_level_values(lvl).unique().values)

    @property
    def all_variables(self):
        lvl = self.var_level_idx
        return self.df.index.get_level_values(lvl).unique().values
    
    def init_glob_widgets(self):
        self.disp_table = ipw.Output()
        self.output = ipw.Output()
        
        btn_clear_output = ipw.Button(description="Clear output",
                                     layout=ipw.Layout(width=self._btn_width))
        btn_clear_output.on_click(self.on_clear_output)
        
        btn_reset = ipw.Button(description="Reset", 
                               tooltip="Reset all changes that were applied",
                               layout=ipw.Layout(width=self._btn_width))
        btn_reset.on_click(self.on_reset)        
        
        tip = ("Save file in {} using filename specified in line above. "
               "Allowed filetypes are: {}".format(self.save_dir, 
                                       list(self.saveas_funs.keys())))
        
        btn_save = ipw.Button(description="Save", 
                              tooltip=tip,
                              layout=ipw.Layout(width=self._btn_width))
        btn_save.on_click(self.on_save)
        
        btn_saveas = ipw.Button(description="Save as", 
                                tooltip="Save current Dataframe as file",
                                layout=ipw.Layout(width=self._btn_width))
        
        btn_plot = ipw.Button(description="Plot",
                              layout=ipw.Layout(width=self._btn_width))
        btn_plot.on_click(self.on_plot)
        
        
        btn_saveas.style.button_color = 'lightgreen'
        btn_saveas.on_click(self.on_saveas)
        
        self.btn_saveas = btn_saveas
        self.glob_toolbar = ipw.HBox([btn_clear_output, 
                                      btn_reset, 
                                      btn_save,
                                      btn_saveas,
                                      btn_plot])
    
        self.save_name = ipw.Text(placeholder='Insert save filename (e.g. test.csv)')
        

    def init_layout(self):
        
        self.edit_ui = ipw.Tab()
        
        self.edit_ui.children = [self.layout_rename, 
                                 self.layout_varselect, 
                                 self.layout_reshaper]
        self.edit_ui.set_title(0, "Rename run")
        self.edit_ui.set_title(1, "Select variables")
        self.edit_ui.set_title(2, "Reshape dataframe")
        
        
        
        self.layout = ipw.VBox([self.edit_ui,
                                self.save_name,
                                self.glob_toolbar,
                                self.disp_table, 
                                self.output], 
                                layout = ipw.Layout(min_height="600px"))
# =============================================================================
#         self.layout.children = [self.layout_varselect, 
#                                 self.layout_rename, 
#                                 self.layout_reshape,
#                                 self.layout_display]
#         
# =============================================================================
    def init_widgets_renamer(self):
        
        self.btn_apply_rename = ipw.Button(description='Apply')
        self.btn_apply_rename.style.button_color = "lightgreen"
        self.btn_apply_rename.on_click(self.on_click_apply_rename)
        self.input_rows_rename = []
        self.input_fields_rename = []
        
        for i, name in enumerate(self.run_names):
            try:
                val = self.new_run_names_init[i]
            except:
                val = name
            ipt = ipw.Text(value=val, placeholder='Insert new name',
                            disabled=False, layout=ipw.Layout(width='100px'))
            row = ipw.HBox([ipw.Label(name, layout=ipw.Layout(width='100px')), ipt])
            self.input_fields_rename.append(ipt)
            self.input_rows_rename.append(row)
        
        self._buttons_edit_df.extend([self.btn_apply_rename])
        
    def init_layout_renamer(self):
        self.layout_rename = ipw.HBox([ipw.VBox(self.input_rows_rename), 
                                       self.btn_apply_rename])
        
    def init_widgets_varselect(self):
        # Init all widgets for variable selector
        self.btn_unselect_all = ipw.Button(description='Unselect all')
        self.btn_select_all = ipw.Button(description='Select all')
        self.btn_flagged = ipw.Button(description="Flagged")
        self.btn_apply_varselect = ipw.Button(description='Apply')
        self.btn_apply_varselect.style.button_color = 'lightgreen'

        self.var_selector = ipw.SelectMultiple(description='', 
                                               options=self.all_variables, 
                                               value=self.default_selection, 
                                               layout=self._base_layout)
        
        self.var_selector_disp = ipw.Textarea(value='', 
                                         description='', 
                                         disabled=True, 
                                         layout=self._base_layout)
        
        
        self.group_selector = ipw.Dropdown(options=self.groups,
                                           value=self.default_selection,
                                            description='',
                                            disabled=False)
        
        # init all actions for widgets of variable selector
        self.var_selector.observe(self.current_varselection)
        self.group_selector.observe(self.on_change_dropdown)
        #what happens when buttons are clicked
        self.btn_select_all.on_click(self.on_select_all_vars_clicked)
        self.btn_unselect_all.on_click(self.on_unselect_all_vars_clicked)
        self.btn_apply_varselect.on_click(self.on_click_apply_varselect)
        
        self._buttons_edit_df.extend([self.btn_select_all,
                                      self.btn_unselect_all,
                                      self.btn_apply_varselect])
        
    def init_layout_varselect(self):
        self.btns_varselect = ipw.VBox([self.btn_select_all, 
                                        self.btn_unselect_all,
                                        ipw.Label(),
                                        self.btn_apply_varselect])
        l = ipw.HBox([ipw.VBox([ipw.Label("Predefined"), self.group_selector]),
                      ipw.VBox([ipw.Label("Index level {}".format(self.var_level_idx)), 
                                self.var_selector]), 
                      ipw.VBox([ipw.Label("Current selection"), 
                                self.var_selector_disp]), 
                      self.btns_varselect])
        self.layout_varselect = l
        
        self.current_varselection(1)
        
        #self.layout = ipw.VBox([self.edit_area, self.output])
        
    def init_layout_reshaper(self):
        
        # COLUMN TO INDEX
        col2idx_header = ipw.Label("Column to index")
        self.col2idx_select =  ipw.SelectMultiple(description='', 
                                                  options=self.column_names, 
                                                  value=(), 
                                                  layout=self._base_layout)
        col2idx_btn_apply = ipw.Button(description = "Add", layout=ipw.Layout(width=self._btn_width))
        col2idx_btn_apply.on_click(self.on_add_col)
        col2idx_btn_apply.tooltip = "Add selected columns to Multiindex"
        col2idx_btn_apply.style.button_color = 'lightgreen'
        
        col2idx_layout = ipw.VBox([col2idx_header,
                                   self.col2idx_select,
                                   ipw.HBox([col2idx_btn_apply])])
        
        # UNSTACKING
        unstack_header = ipw.Label("Unstack index")
        self.unstack_select =  ipw.SelectMultiple(description='', 
                                                  options=self.index_level_names, 
                                                  value=(), 
                                                  layout=self._base_layout)
        unstack_btn_apply = ipw.Button(description = "Apply", layout=ipw.Layout(width=self._btn_width))
        unstack_btn_apply.on_click(self.on_unstack)
        unstack_btn_apply.style.button_color = 'lightgreen'
        unstack_btn_apply.tooltip = "Put selected indices into columns"
        
        unstack_layout = ipw.VBox([unstack_header,
                                   self.unstack_select,
                                   ipw.HBox([unstack_btn_apply])])
        
        
        # STACKING
        stack_header = ipw.Label("Stack index")
        self.stack_select =  ipw.SelectMultiple(description='', 
                                                  options=self.index_level_col_names,
                                                  value=(), 
                                                  layout=self._base_layout)
        stack_btn_apply = ipw.Button(description = "Apply", layout=ipw.Layout(width=self._btn_width))
        stack_btn_apply.on_click(self.on_stack)
        stack_btn_apply.style.button_color = 'lightgreen'
        stack_btn_apply.tooltip = "Put selected indices into rows"
        
        stack_layout = ipw.VBox([stack_header,
                                 self.stack_select,
                                 ipw.HBox([stack_btn_apply])])
        # SELECT COLUMN
        extract_header = ipw.Label("Extract column")
        self.extract_select =  ipw.Select(description='', 
                                          options=self.data_column_names,
                                          layout=self._base_layout)
        
        extract_btn_apply = ipw.Button(description="Apply", 
                                       layout=ipw.Layout(width=self._btn_width))
        extract_btn_apply.on_click(self.on_extract)
        extract_btn_apply.style.button_color = 'lightgreen'
        extract_btn_apply.tooltip = "Extract currently selected column"
        
        extract_btn_undo = ipw.Button(description="Undo", 
                                     layout=ipw.Layout(width=self._btn_width))
        extract_btn_undo.on_click(self.on_extract_undo)
        extract_btn_undo.tooltip = "Undo last column extraction"
        
        extract_layout = ipw.VBox([extract_header,
                                   self.extract_select,
                                   ipw.HBox([extract_btn_undo,
                                             extract_btn_apply])])
        
        self.layout_reshaper = ipw.HBox([col2idx_layout, 
                                         unstack_layout, 
                                         stack_layout,
                                         extract_layout])
        
        self._buttons_edit_df.extend([col2idx_btn_apply,
                                      unstack_btn_apply,
                                      stack_btn_apply,
                                      extract_btn_apply])
    # Methods for renamer
    def on_click_apply_rename(self, b):
        self.apply_changes_rename()
        self.disp_current()   
        
    def apply_changes_rename(self):
        
        df = self.df_edit
        mapping = od()
        
        for i, name in enumerate(self.run_names):
            repl = str(self.input_fields_rename[i].value)
            mapping[name] = repl
        self.df_edit = df.rename(index=mapping, level=self.run_level_idx)
        self.output.append_display_data("Applying renaming: {}".format(mapping))
    # Methods for variable selector
    def on_unselect_all_vars_clicked(self, b):
        self.unselect_all()
    
    def on_select_all_vars_clicked(self, b):
        self.select_all()
    
    def on_change_dropdown(self, b):
        self.select_current_group()
        
    def unselect_all(self):
        self.var_selector.value = ()
    
    def select_all(self):
        self.var_selector.value = self.var_selector.options
    
    def select_current_group(self):
        self.var_selector.value = self.group_selector.value
    
    def current_varselection(self, b):
        s=""
        for item in self.var_selector.value:
            s += "{}\n".format(item)
        self.var_selector_disp.value = s
        
    def crop_var_selection(self):
        try:
            self.df_edit = helpers.crop_selection_dataframe(self.df_edit, 
                                                            self.var_selector.value, 
                                                            levels=self.var_level_idx)
            self.output.append_display_data("Applying variable selection: {}".format(self.var_selector.value))
        except Exception as e:
            self.output.append_display_data("WARNING: failed to extract selection.\nTraceback {}".format(format_exc()))
    
    def on_click_apply_varselect(self, b):
        self.crop_var_selection()
        self.disp_current()
        
    # Methods for reshaper
    def update_ui(self):
        """Recreate user interface"""
        if not isinstance(self.df_edit, pd.Series):
                
            if isinstance(self.df_edit.columns, pd.MultiIndex):
                self.col2idx_select.options = ("N/A", "Current dataframe is unstacked")
                self.col2idx_select.disabled = True
                for item in self.input_fields_rename:
                    item.disabled = True
                self.btn_apply_rename.disabled=True
                tip = ("Dataframe contains unstacked indices. Renaming can only be "
                       "applied for dataframe that has not been unstacked. You "
                       "may re-stack the dataframe using the tab 'Reshape dataframe'")
                self.btn_apply_rename.tooltip = tip
                self.btn_apply_varselect.disabled = True
                self.btn_apply_varselect.tooltip = tip
            else:
                self.col2idx_select.options = self.column_names
                self.col2idx_select.value=()
                self.col2idx_select.disabled = False
                for item in self.input_fields_rename:
                    item.disabled = False
                self.btn_apply_rename.disabled=False
                self.btn_apply_varselect.disabled=False
                tip = ("Apply current settings")
                self.btn_apply_rename.tooltip = tip
                self.btn_apply_varselect.tooltip = tip
            
            self.unstack_select.options = self.index_level_names
            self.unstack_select.value = ()
            
            self.stack_select.options = self.index_level_col_names
            self.stack_select.value = ()
            
            self.extract_select.options = self.data_column_names
            
        self.disp_table.clear_output()
        self.disp_current()
        
    def on_add_col(self, b):
        var_names = list(self.col2idx_select.value)
        self.add_to_index(var_names)
        self.update_ui()
    
    def on_unstack(self, b):
        level_names = list(self.unstack_select.value)
        self.unstack(level_names)
        self.update_ui()
        
    def on_stack(self, b):
        level_names = list(self.stack_select.value)
        self.stack(level_names)
        self.update_ui()
       
    def on_extract(self, b):
        val = str(self.extract_select.value)
        self._df_edit_last = self.df_edit
        self.df_edit = self.df_edit[val]
        self.update_ui()
        self.freeze_ui()
        self.disp_current()
        
    def freeze_ui(self, disable=True):
        for btn in self._buttons_edit_df:
            btn.disabled = disable
            
    def on_extract_undo(self, b):
        self.df_edit = self._df_edit_last
        self.update_ui()
        self.freeze_ui(False)
        self.disp_current()
        
    # global events
    def on_clear_output(self, b):
        self.output.clear_output()
        
    def on_save(self, b):
        self.save()
        
    def on_saveas(self, b):
        self.saveas()
        
    def on_reset(self, b):
        self.reset()
        self.update_ui()
    
    def on_plot(self, b):
        self.plot()

    def check_shape_init(self, df):
        if isinstance(df.columns, pd.MultiIndex):
            #print("Initial Dataframe is unstacked, stacking back")
            return helpers.stack_dataframe_original_idx(df)
        return deepcopy(df)
    
    def add_to_index(self, var_names):
        if isinstance(var_names, str):
            var_names = [var_names]
        for item in var_names:
            self.df_edit = self.df_edit.set_index([self.df_edit.index, item])
    
    def unstack(self, level_names):
        self.df_edit = self.df_edit.unstack(level_names)
        
    def stack(self, level_names):
        self.df_edit = helpers.stack_dataframe(self.df_edit, level_names)
        
    def reset(self):
        self.df_edit = self.check_shape_init(self.df)
          
    def disp_current(self):
        #self.output.append_display_data(ipw.Label("PREVIEW current selection", fontsize=22))
        self.disp_table.clear_output()
        if isinstance(self.df_edit, pd.Series):
            disp = self.df_edit
        else:
            disp = self.df_edit.head().style.set_caption("PREVIEW")
        self.disp_table.append_display_data(disp)
        #self.disp_table.append_display_data(preview)
        #self.output
       
    def plot_heatmap(self, ax):
        try:
            self.current_plot = helpers.df_to_heatmap(self.df_edit, ax=ax, 
                                                      **self.heatmap_settings)
        except Exception as e:
            self.output.append_display_data("Failed to plot heatmap: Error "
                                            "message: {}".format(repr(e)))
        
    def plot(self):
        self.disp_table.clear_output()
        with self.disp_table:
            fig, ax = plt.subplots(1,1, figsize=(14, 8))
            self.plot_heatmap(ax=ax)
            plt.show()
            
            #self.default_plot_fun()
        #self.disp_table.append_display_data()
    def save_png(self, fpath):
        if not self.current_plot:
            self.default_plot_fun()
        self.current_plot.figure.savefig(fpath)
        
    def save_csv(self, fpath):
        self.df_edit.to_csv(fpath)
    
    def save_xlsx(self, fpath):
        writer = pd.ExcelWriter(fpath)
        self.df_edit.to_excel(writer)
        writer.save()
        writer.close()
        
    def open_saveas_dialog(self):
        """Generate instance of tkinter.asksaveasfilename
        """
        # Create Tk root
        root = Tk()
        # Hide the main window
        root.withdraw()
        # Raise the root to the top of all windows.
        root.call('wm', 'attributes', '.', '-topmost', True)
        # List of selected fileswill be set to b.value
        filename = filedialog.asksaveasfilename(initialdir=self.save_dir,
                                                title = "Save as",
                                                filetypes = (("csv files","*.csv"),
                                                             ("Excel files","*.xlsx"),
                                                             ("PNG files", "*.png")))
        return filename
        
    def save(self):
        savename = os.path.basename(self.save_name.value)
        self.saveas(filename=os.path.join(self.save_dir, savename))
        
    def saveas(self, filename=None):
        msg = "Failed to save file, enter valid filename and type"
        if filename is None:
            if tkinter_available:
                filename = self.open_saveas_dialog()
            else:
                msg = ("Failed to save table. Could not open file dialog "
                       "please install tkinter or insert valid name in "
                       "line above")
        
        for ftype, func in self.saveas_funs.items():
            if filename.lower().endswith(ftype):
                try:
                    func(filename)
                    msg = "Succesfully saved: {}".format(filename)
                except Exception as e:
                    msg = ("Failed to save {}. Error {}".format(filename, repr(e)))
                break
            
        self.output.append_display_data(msg)
        
                    
    def __call__(self):
        return self.layout
    
class IndexRenamer(object):
    output = ipw.Output()
    def __init__(self, df, level=0, suggestions=[]):
        self.df = df
        self._df_edit = df
        self.level = level
        
        self.suggestions = suggestions
      
        self.init_widgets()
        self.init_actions()
        self.init_layout()
        
        self.renamed_info = od()
        self.apply_changes()
        
    @property
    def names(self):
        #return sorted(self.df.index.get_level_values(self.level).unique().values)
        return self.df.index.get_level_values(self.level).unique().values
    @property
    def df_edit(self):
        return deepcopy(self._df_edit)
    
    def init_widgets(self):
        
        self.btn_apply = ipw.Button(description='Apply')
        self.btn_apply.style.button_color = "lightgreen"
        
        self.input_rows = []
        self.input_fields = []
        
        for i, name in enumerate(self.names):
            try:
                val = self.suggestions[i]
            except:
                val = name
            ipt = ipw.Text(value=val, placeholder='Insert new name',
                            disabled=False, layout=ipw.Layout(width='100px'))
            row = ipw.HBox([ipw.Label(name, layout=ipw.Layout(width='100px')), ipt])
            self.input_fields.append(ipt)
            self.input_rows.append(row)
                                      
    def init_actions(self):
        #what happens when the state of the selection is changed (display current selection)
        self.btn_apply.on_click(self.on_click_apply)
        
    def init_layout(self):
        
        
        edit_area = ipw.HBox([ipw.VBox(self.input_rows), self.btn_apply])

        self.layout = ipw.VBox([edit_area, self.output])

        
    def on_click_apply(self, b):
        self.apply_changes()
        
    def disp_current(self):
        self.output.clear_output()
        #self.output.append_display_data(ipw.Label("PREVIEW current selection", fontsize=22))
        self.output.append_display_data(self._df_edit.style.set_caption("PREVIEW"))
        self.output
        
    def apply_changes(self):
        
        df = self.df 
        mapping = od()
        
        for i, name in enumerate(self.names):
            repl = str(self.input_fields[i].value)
            mapping[name] = repl
        self._df_edit = df.rename(index=mapping, level=self.level)
        
        self.disp_current()
        
    def __call__(self):
        return self.layout
    
class SelectVariable(object):
    output = ipw.Output()
    def __init__(self, df, level, preconfig_file=None, default_group=None):
        #df.sort_index(inplace=True)
        self.df = df
        self._df_edit = df
        self.level = level
        
        self.groups = od()
        self.groups["flagged"] = self.flagged_vars
        if preconfig_file:
            self.groups.update(helpers.load_varconfig_ini(preconfig_file))
        if default_group is None:
            default_group = "flagged"
        
        if not default_group in self.groups:
            raise ValueError("No default group with ID {} in file {}".format(default_group, preconfig_file))
            
        self.default_selection = self.groups[default_group]
        
        self.vals = self.df.index.get_level_values(self.level).unique().values
        
        
        self._base_layout = ipw.Layout(flex='0 1 auto', 
                                       height='200px', 
                                       min_height='200px', 
                                       width='auto')
    
        self.init_widgets()
        self.init_actions()
        self.init_layout()
        
        self.select_current_group()
        self.print_current(1)
        self.crop_selection()
        self.disp_current()
    
    @property
    def df_edit(self):
        return deepcopy(self._df_edit)
    
    @property
    def flagged_vars(self):
        return list(self.df[self.df.Flag].index.get_level_values(self.level).unique().values)
    
    def init_widgets(self):
        
        self.btn_unselect_all = ipw.Button(description='Unselect all')
        self.btn_select_all = ipw.Button(description='Select all')
        self.btn_flagged = ipw.Button(description="Flagged")
        self.btn_apply = ipw.Button(description='Apply')
        self.btn_apply.style.button_color = 'lightgreen'

        self.var_selector = ipw.SelectMultiple(description='', 
                                               options=self.vals, 
                                               value=self.flagged_vars, 
                                               min_width='150px',
                                               layout=self._base_layout)
        
        self.current_disp = ipw.Textarea(value='', 
                                         description='', 
                                         disabled=True, 
                                         layout=self._base_layout)
        
        #groups = [key for key, val in self.groups.items()]
        
        self.group_selector = ipw.Dropdown(options=self.groups,
                                           value=self.default_selection,
                                            description='',
                                            disabled=False)
        #self.output = ipw.Output()
        
    def init_actions(self):
        #what happens when the state of the selection is changed (display current selection)
        self.var_selector.observe(self.print_current)
        self.group_selector.observe(self.on_change_dropdown)
        #what happens when buttons are clicked
        self.btn_select_all.on_click(self.on_select_all_vars_clicked)
        self.btn_unselect_all.on_click(self.on_unselect_all_vars_clicked)
        self.btn_apply.on_click(self.on_click_apply)
    
    def init_layout(self):
        
        self.btns = ipw.VBox([self.btn_select_all, 
                              self.btn_unselect_all,
                              ipw.Label(),
                              self.btn_apply])
    
        self.edit_area = ipw.HBox([ipw.VBox([ipw.Label("Predefined"), self.group_selector]),
                                   ipw.VBox([ipw.Label("Index level {}".format(self.level)), self.var_selector]), 
                                   ipw.VBox([ipw.Label("Current selection"), self.current_disp]), 
                                   self.btns])
        
        self.layout = ipw.VBox([self.edit_area, self.output])
    
    def on_unselect_all_vars_clicked(self, b):
        self.unselect_all()
    
    def on_select_all_vars_clicked(self, b):
        self.select_all()
    
    def on_change_dropdown(self, b):
        self.select_current_group()
        
    def unselect_all(self):
        self.var_selector.value = ()
    
    def select_all(self):
        self.var_selector.value = self.var_selector.options
    
    def select_current_group(self):
        self.var_selector.value = self.group_selector.value
        
    def disp_current(self):
        self.output.clear_output()
        #self.output.append_display_data(ipw.Label("PREVIEW current selection", fontsize=22))
        self.output.append_display_data(self._df_edit.head().style.set_caption("PREVIEW HEAD"))
        self.output
        
    def crop_selection(self):
        try:
            self._df_edit = helpers.crop_selection_dataframe(self.df, 
                                                             self.var_selector.value, 
                                                             levels=self.level)
        except Exception as e:
            print("WARNING: failed to extract selection.\nTraceback {}".format(format_exc()))
    
    def on_click_apply(self, b):
        self.crop_selection()
        self.disp_current()
        
    def print_current(self, b):
        s=""
        for item in self.var_selector.value:
            s += "{}\n".format(item)
        self.current_disp.value = s
    
    def __repr__(self):
        return repr(self.layout)
    
    def __call__(self):
        return self.layout
    
    
class EditDictCSV(object):
    """Widget that can be used to interactively edit a CSV file
    
    The CSV is supposed to be created from a "simple" dictionary with entries
    strings.
    """
    output = ipw.Output()
    def __init__(self, csv_loc):
        self.csv_loc = csv_loc
        self.load_csv()
            
        self.init_widgets()
        self.init_actions()
        self.init_layout()
    
    def init_widgets(self):
        
        self.btn_update = ipw.Button(description='Update',
                                     tooltip=('Updates the current dictionary based on values in text fields'
                                              '(for further analysis, use Save csv button to write to CSV)'))
        self.btn_reload = ipw.Button(description='Reload',
                                     tooltip='Reloads information from file var_info.csv')
        self.btn_save = ipw.Button(description='Update and save',
                                     tooltip='Updates current selection and writes to CSV')
        
        self.btn_save.style.button_color = "lightgreen"
        
        self.input_rows = []
        self.input_fields = {}
        
        for name,  val in self.var_dict.items():
            ipt = ipw.Text(value=val, placeholder='Insert new name',
                            disabled=False, 
                            layout = ipw.Layout(min_width="200px"))
            row = ipw.HBox([ipw.Label(name, layout=ipw.Layout(min_width="200px")), ipt])
            self.input_fields[name] = ipt
            self.input_rows.append(row)  
            
    def init_actions(self):
        self.btn_update.on_click(self.on_click_update)
        self.btn_reload.on_click(self.on_click_load_csv)
        self.btn_save.on_click(self.on_click_save)
        
    def init_layout(self):
        
        vbox_buttons = ipw.VBox([self.btn_reload,
                                 self.btn_update,
                                 self.btn_save])
        self.layout = ipw.HBox([ipw.VBox(self.input_rows), vbox_buttons, 
                                self.output])
        
    def on_click_update(self, b):
        self.apply_changes()
    
    def on_click_load_csv(self, b):
        self.load_csv()
        self.update_info_fields()
        
    def on_click_save(self, b):
        self.save_csv()
    
    def save_csv(self):
        self.apply_changes()
        helpers.save_varinfo_dict_csv(self.var_dict, self.csv_loc)
        
    def load_csv(self):
        if self.csv_loc is None or not os.path.exists(self.csv_loc):
            raise IOError("Please provide path to csv file")
        try:
            self.var_dict = helpers.load_varinfo_dict_csv(self.csv_loc)
        except Exception as e:
            self.write_to_output(format_exc())
    
    def update_info_fields(self):
        for key, val in self.var_dict.items():
            self.input_fields[key].value = val
    
    def write_to_output(self, msg):
        self.output.append_display_data(msg)
        self.output
        
    def apply_changes(self):
        
        new = od()
        for key, edit in self.input_fields.items():
            new[key] = edit.value
        
        self.var_dict = new
        
    def __call__(self):
        return self.layout
    
class SaveAsButton(ipw.Button):
    """A file widget that leverages tkinter.filedialog.
    
    Based on and modified from ``SelectFilesButton`` (see below) or here: 
        
    https://codereview.stackexchange.com/questions/162920/file-selection-button-for-jupyter-notebook
    
    """

    def __init__(self, save_dir=None):
        super(SaveAsButton, self).__init__()
        # Add the selected_files trait
        self.add_traits(files=traitlets.traitlets.List())
        
        if not save_dir:
            save_dir = os.getcwd()
        self.save_dir = save_dir
        # Create the button.
        self.description = "Save as"
        self.icon = "square-o"
        self.style.button_color = "orange"
        self.file_name = ""
        # Set on click behavior.
        self.on_click(self.saveas)


    def saveas(self, b):
        """Generate instance of tkinter.asksaveasfilename

        Parameters
        ----------
        b : obj:
            An instance of ipywidgets.widgets.Button 
        """
        # Create Tk root
        root = Tk()
        # Hide the main window
        root.withdraw()
        # Raise the root to the top of all windows.
        root.call('wm', 'attributes', '.', '-topmost', True)
        # List of selected fileswill be set to b.value
        self.file_name = filedialog.asksaveasfilename(initialdir=self.save_dir,
                                                      title = "Save as",
                                                      filetypes = (("csv files","*.csv"),("all files","*.*")))

        self.description = "Files Selected"
        self.icon = "check-square-o"
        self.style.button_color = "lightgreen"   

### DOWNLOADED
        
class SelectFilesButton(ipw.Button):
    """A file widget that leverages tkinter.filedialog.
    
    
    Downloaded here: https://codereview.stackexchange.com/questions/162920/file-selection-button-for-jupyter-notebook
    
    """

    def __init__(self):
        super(SelectFilesButton, self).__init__()
        # Add the selected_files trait
        self.add_traits(files=traitlets.traitlets.List())
        # Create the button.
        self.description = "Select Files"
        self.icon = "square-o"
        self.style.button_color = "orange"
        # Set on click behavior.
        self.on_click(self.select_files)

    @staticmethod
    def select_files(b):
        """Generate instance of tkinter.filedialog.

        Parameters
        ----------
        b : obj:
            An instance of ipywidgets.widgets.Button 
        """
        # Create Tk root
        root = Tk()
        # Hide the main window
        root.withdraw()
        # Raise the root to the top of all windows.
        root.call('wm', 'attributes', '.', '-topmost', True)
        # List of selected fileswill be set to b.value
        b.files = filedialog.askopenfilename(multiple=True)

        b.description = "Files Selected"
        b.icon = "check-square-o"
        b.style.button_color = "lightgreen"
        
class FileBrowser(object):
    """Widget for browsing files interactively
    
    The widget was downloaded and modified from here:
        
        https://gist.github.com/DrDub/6efba6e522302e43d055#file-selectfile-py
        
    """
    def __init__(self, path=None):
        if path is None:
            path = os.getcwd()
        self.path = path
        self._update_files()
        
    def _update_files(self):
        self.files = list()
        self.dirs = list()
        if(os.path.isdir(self.path)):
            for f in os.listdir(self.path):
                ff = self.path + "/" + f
                if os.path.isdir(ff):
                    self.dirs.append(f)
                else:
                    self.files.append(f)
                    
    def _update(self, box):
        
        def on_click(b):
            if b.description == '..':
                self.path = os.path.split(self.path)[0]
            else:
                self.path = self.path + "/" + b.description
            self._update_files()
            self._update(box)
        
        buttons = []
        if self.files:
            button = ipw.Button(description='..', background_color='#d0d0ff')
            button.on_click(on_click)
            buttons.append(button)
        for f in self.dirs:
            button = ipw.Button(description=f, background_color='#d0d0ff')
            button.on_click(on_click)
            buttons.append(button)
        for f in self.files:
            button = ipw.Button(description=f)
            button.on_click(on_click)
            buttons.append(button)
        box.children = tuple([ipw.HTML("<h2>%s</h2>" % (self.path,))] + buttons)
        
    def __call__(self):
        box = ipw.VBox()
        self._update(box)
        return box

### UNDER DEVELOPMENT
class SelectVariableNew(object):
    output = ipw.Output()
    def __init__(self, df, num_cols=3):
        #df.sort_index(inplace=True)
        self.df = df
        #self.vals = tuple(self.df.index.levels[2].values)
        self.vals = self.df.index.get_level_values("Variable").unique().values
        self._df_edit = df
        self.num_cols = num_cols
        self.init_widgets()
        self.init_actions()
        self.init_layout()
        
        #self.print_current(1)
        self.highlight_selection(1)
        #self.crop_selection()
        self.disp_current()
    
    @property
    def df_edit(self):
        return deepcopy(self._df_edit)
    
    @property
    def flagged_vars(self):
        return list(self.df[self.df.Flag].index.get_level_values("Variable").unique().values)
    
    def init_widgets(self):
        
        self.btn_unselect_all = ipw.Button(description='Unselect all')
        self.btn_select_all = ipw.Button(description='Select all')
        self.btn_flagged = ipw.Button(description="Flagged")
        self.btn_apply = ipw.Button(description='Apply')
        self.btn_apply.style.button_color = 'lightgreen'
        
        self.items = []
        self.input_fields = []
    
        for num, name in enumerate(self.vals):
            ipt = ipw.Button(description=name, disabled=True, layout=ipw.Layout(width='150px'))
            order = ipw.Text(value=str(num+1), disabled=False, layout=ipw.Layout(width='50px'))
            order.observe(self.on_change_input_field)
            self.items.append(ipt)
            self.input_fields.append(order)
            
        #self.output = ipw.Output()
    
    def init_actions(self):
        #what happens when the state of the selection is changed (display current selection)
        #self.var_selector.observe(self.print_current)
        #what happens when buttons are clicked
        self.btn_select_all.on_click(self.on_select_all_vars_clicked)
        self.btn_unselect_all.on_click(self.on_unselect_all_vars_clicked)
        self.btn_flagged.on_click(self.on_flagged_clicked)
        self.btn_apply.on_click(self.on_click_apply)
    
        
    def init_layout(self):
        
        self.btns = ipw.HBox([self.btn_select_all, 
                              self.btn_unselect_all,
                              self.btn_flagged,
                              self.btn_apply])
        
        self.init_input_area()
        
        self.layout = ipw.VBox([self.btns, self.edit_area, self.output])
        
    def init_input_area(self):
        
        num_cols = self.num_cols
        items = self.items
        input_fields = self.input_fields
    
        col_vals = np.array_split(np.arange(len(items)), num_cols)
        
        cols = []
        for ival in col_vals:
            col_rows = []
            for val in ival:
                col_rows.append(ipw.HBox([items[val], input_fields[val]]))
            cols.append(ipw.VBox(col_rows))
            
        self.edit_area = ipw.HBox(cols)

    
    def on_unselect_all_vars_clicked(self, b):
        self.unselect_all()
    
    def on_select_all_vars_clicked(self, b):
        self.select_all()
    
    def on_flagged_clicked(self, b):
        self.select_flagged()
        
    def on_change_input_field(self, b):
        print(b.new.value)
        
    @property
    def current_order(self):
        nums = []
        for item in self.input_fields:
            nums.append(item.value)
        
    def highlight_selection(self, b):
        for i, item in enumerate(self.input_fields):
            try:
                int(item.value)
                self.items[i].style.button_color = "#e6ffee"
            except:
                self.items[i].style.button_color = "white"
                
    def unselect_all(self):
        pass
        #self.var_selector.value = ()
    
    def select_all(self):
        pass
        #self.var_selector.value = self.var_selector.options
    
    def select_flagged(self):
        pass
        #self.var_selector.value = self.flagged_vars
        
    def disp_current(self):
        self.output.clear_output()
        #self.output.append_display_data(ipw.Label("PREVIEW current selection", fontsize=22))
        self.output.append_display_data(self._df_edit.head().style.set_caption("PREVIEW HEAD"))
        self.output
        
    def crop_selection(self):
        raise NotImplementedError
    
    def on_click_apply(self, b):
        self.crop_selection()
        self.disp_current()
    
    def __repr__(self):
        return repr(self.layout)
    
    def __call__(self):
        return self.layout
    
class ReshapeAndSelect(object):
    """Widget that can be used to reshape a Dataframe and select individual data columns"""
    output = ipw.Output()
    def __init__(self, df):
        
        raise NotImplementedError()
        self.df = df
        self._df_edit = df
        
        self.index_names = df.index.names
        self.col_names = df.columns
    
    @property
    def df_edit(self):
        return deepcopy(self._df_edit)
    
    @property
    def flagged_vars(self):
        return list(self.df[self.df.Flag].index.get_level_values("Variable").unique().values)
    
    def init_widgets(self):
    
        self.btn_unselect_all = ipw.Button(description='Unselect all')
        self.btn_select_all = ipw.Button(description='Select all')
        self.btn_flagged = ipw.Button(description="Flagged")
        self.btn_apply = ipw.Button(description='Apply')
        self.btn_apply.style.button_color = 'lightgreen'

        self.var_selector = ipw.SelectMultiple(description="Variables", 
                                               options=self.vals, 
                                               value=self.flagged_vars, 
                                               min_width='150px',
                                               layout=self.box_layout)
        
        self.current_disp = ipw.Textarea(value='', 
                                         description='Current:', 
                                         disabled=True, 
                                         layout=self.box_layout)
        #self.output = ipw.Output()
        
    def init_actions(self):
        #what happens when the state of the selection is changed (display current selection)
        self.var_selector.observe(self.print_current)
        #what happens when buttons are clicked
        self.btn_select_all.on_click(self.on_select_all_vars_clicked)
        self.btn_unselect_all.on_click(self.on_unselect_all_vars_clicked)
        self.btn_flagged.on_click(self.on_flagged_clicked)
        self.btn_apply.on_click(self.on_click_apply)
    
    def init_display(self):
        self.btns = ipw.VBox([self.btn_select_all, 
                              self.btn_unselect_all,
                              self.btn_flagged,
                              ipw.Label(),
                              self.btn_apply])
    
        self.edit_area = ipw.HBox([self.var_selector, 
                                   self.current_disp, 
                                   self.btns])
        
        self.layout = ipw.VBox([self.edit_area, self.output])
    
    def on_unselect_all_vars_clicked(self, b):
        self.unselect_all()
    
    def on_select_all_vars_clicked(self, b):
        self.select_all()
    
    def on_flagged_clicked(self, b):
        self.select_flagged()
        
    def unselect_all(self):
        self.var_selector.value = ()
    
    def select_all(self):
        self.var_selector.value = self.var_selector.options
    
    def select_flagged(self):
        self.var_selector.value = self.flagged_vars
        
    def disp_current(self):
        self.output.clear_output()
        #self.output.append_display_data(ipw.Label("PREVIEW current selection", fontsize=22))
        self.output.append_display_data(self._df_edit.head().style.set_caption("PREVIEW HEAD"))
        self.output
        
    def crop_selection(self):
        idx = pd.IndexSlice
        try:
            self._df_edit = self.df.loc[idx[:, :, self.var_selector.value, :], :]
        except Exception as e:
            print("WARNING: failed to extract selection.\nTraceback {}".format(format_exc()))
    
    def on_click_apply(self, b):
        self.crop_selection()
        self.disp_current()
        
    def print_current(self, b):
        s=""
        for item in self.var_selector.value:
            s += "{}\n".format(item)
        self.current_disp.value = s
    
    def __repr__(self):
        return repr(self.layout)
    
    def __call__(self):
        return self.layout
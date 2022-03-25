from IPython.display import display, Markdown, clear_output
import ipywidgets as widgets

import shutil
import os

import IPython
from ipywidgets import widgets

from ipywidgets import Layout, Button, Box  

from ipywidgets import AppLayout, Button, Layout

from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
#-----------------------------------------#

#-----------------------------------------#
#            PARTIE HUGO                  #
#-----------------------------------------#

def moveFile(fileName, pathToMove):      
       
    try: 
        #move the file in inpute
        shutil.move(fileName, pathToMove)
        print("File is moved successfully")  
        
    # If Source is a file but destination is a directory
    except IsADirectoryError:
        print("Source is a file but destination is a directory.")
    
    # If source is a directory but destination is a file
    except NotADirectoryError:
        print("Source is a directory but destination is a file.")
    
    # For permission related errors
    except PermissionError:
        print("Operation not permitted.")

    # For other errors
    except OSError as error:
        print(error)
        

        # Implementation de la fonction
        # Au niveau des boutons ci-dessous
        
        
#-----------------------------------------#
#            PARTIE HUGO/                 #
#-----------------------------------------#
        
#-----------------------------------------#

#-----------------------------------------#
#            PARTIE MARION                #
#-----------------------------------------#


url = 'detected.jpg'
image = IPython.display.Image(url, width = max)

boximg = widgets.Image(
  value=image.data,
  format='jpg', 
  layout=widgets.Layout(border='3px solid black')
)
#-----------------------------------------#
urlG = input()
path = os.path.abspath(urlG)
image = IPython.display.Image(urlG, width = max)

#-----------------------------------------#
#-----------------------------------------#
#-----------------------------------------#
button1 = widgets.Button(description='1')
out = widgets.Output()
def on_button_clicked(_):
      with out:
          path = r"C:\Users\hugo\Documents\Filippi _Coléoptère\1"
          moveFile(url, path)
button1.on_click(on_button_clicked)
widgets.VBox([button1,out])
#-----------------------------------------#
button2 = widgets.Button(description='2')
out = widgets.Output()
def on_button_clicked(_):
      with out:
          path = r"C:\Users\hugo\Documents\Filippi _Coléoptère\2"
          moveFile(urlG, path)
button2.on_click(on_button_clicked)
widgets.VBox([button2,out])
#-----------------------------------------#
button3 = widgets.Button(description='3')
out = widgets.Output()
def on_button_clicked(_):
      with out:
          path = r"C:\Users\hugo\Documents\Filippi _Coléoptère\3"
          moveFile(urlG, path )
button3.on_click(on_button_clicked)
widgets.VBox([button3,out])
#-----------------------------------------#
button4 = widgets.Button(description='4')
out = widgets.Output()
def on_button_clicked(_):
      with out:
          path = r"C:\Users\hugo\Documents\Filippi _Coléoptère\4"
          moveFile(urlG, path)
button4.on_click(on_button_clicked)
widgets.VBox([button4,out])
#-----------------------------------------#
button5 = widgets.Button(description='5')
out = widgets.Output()
def on_button_clicked(_):
      with out:
          path = r"C:\Users\hugo\Documents\Filippi _Coléoptère\5"
          moveFile(urlG, path)
button5.on_click(on_button_clicked)
widgets.VBox([button5,out])
#-----------------------------------------#
button6 = widgets.Button(description='6')
out = widgets.Output()
def on_button_clicked(_):
      with out:
          path = r"C:\Users\hugo\Documents\Filippi _Coléoptère\6"
          moveFile(urlG, path)
button6.on_click(on_button_clicked)
widgets.VBox([button6,out])
#-----------------------------------------#
button7 = widgets.Button(description='7')
out = widgets.Output()
def on_button_clicked(_):
      with out:
          path = r"C:\Users\hugo\Documents\Filippi _Coléoptère\7"
          moveFile(urlG, path)
button7.on_click(on_button_clicked)
widgets.VBox([button7,out])
#-----------------------------------------#
button8 = widgets.Button(description='8')
out = widgets.Output()
def on_button_clicked(_):
      with out:
          path = r"C:\Users\hugo\Documents\Filippi _Coléoptère\8"
          moveFile(urlG, path)
button8.on_click(on_button_clicked)
widgets.VBox([button8,out])
#-----------------------------------------#
button9 = widgets.Button(description='9')
out = widgets.Output()
def on_button_clicked(_):
      with out:
          path = r"C:\Users\hugo\Documents\Filippi _Coléoptère\9"
          moveFile(urlG, path)
button9.on_click(on_button_clicked)
widgets.VBox([button9,out])
#-----------------------------------------#
button10 = widgets.Button(description='10')
out = widgets.Output()
def on_button_clicked(_):
      with out:
          path = r"C:\Users\hugo\Documents\Filippi _Coléoptère\10"
          moveFile(urlG, path)
button10.on_click(on_button_clicked)
widgets.VBox([button10,out])
#-----------------------------------------#
#-----------------------------------------#
buttonAUTRE = widgets.Button(description='none')
out = widgets.Output()
def on_button_clicked(_):
      with out:
          path = r"C:\Users\hugo\Documents\Filippi _Coléoptère\none"
          moveFile(urlG, path)
buttonAUTRE.on_click(on_button_clicked)
widgets.VBox([buttonAUTRE,out])
#-----------------------------------------#
#-----------------------------------------#
buttonPDR = widgets.Button(description='no_recept')
out = widgets.Output()
def on_button_clicked(_):
      with out:
          path = r"C:\Users\hugo\Documents\Filippi _Coléoptère\no_recept"
          moveFile(urlG, path)
buttonPDR.on_click(on_button_clicked)
widgets.VBox([buttonPDR,out])

#-----------------------------------------#
#-----------------------------------------#
#-----------------------------------------#



boximgG = widgets.Image(
  value=image.data,
  format='jpg', 
  width=650,
  height=750,
  layout=widgets.Layout(border='3px solid black'),
)
lblIMGfirst=widgets.Label()
lblIMG=widgets.Label()
lblIMG.value = urlG
lblIMGfirst.value = url
#-----------------------------------------#

box12 = widgets.HBox([button1, button2])
box34 = widgets.HBox([button3, button4])
box56 = widgets.HBox([button5, button6])
box78 = widgets.HBox([button7, button8])
box910 = widgets.HBox([button9, button10])
boxAUTREPDR = widgets.HBox([buttonAUTRE, buttonPDR])
boxBouton = widgets.VBox([box12,box34,box56,box78,box910,boxAUTREPDR])
boxImgG = widgets.VBox([lblIMGfirst, boximgG])
boxImg = widgets.VBox([lblIMG, boximg])

#-----------------------------------------#

boxDroite = widgets.VBox([boxImg, boxBouton],layout=widgets.Layout(display="flex",align_items="stretch"))

#-----------------------------------------#
BOX = widgets.AppLayout(
                       width="100%",
                       height = "100%",
                       header=None,   
                       left_sidebar=boxImgG,
                       center=None,
                       right_sidebar=boxDroite,
                       footer=None,
                        pane_widths=[2,0,1]
                      )
#-----------------------------------------#
#-----------------------------------------#
#-----------------------------------------#


#-----------------------------------------#
#            PARTIE MARION/               #
#-----------------------------------------#

        
BOX
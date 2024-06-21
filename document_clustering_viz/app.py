from shiny import App, render, ui
from os import walk
from pathlib import Path

dir = Path(__file__).resolve().parent
filenames = next(walk(dir / "results/images/"), (None, None, []))[2]  # [] if no file
filenames2 = next(walk(dir / "results/images/sub_images/"), (None, None, []))[2]  # [] if no file

app_ui = ui.page_sidebar(  
    ui.sidebar("Sidebar", bg="#f8f8f8",
               open="desktop"
               ),  
    ui.layout_columns(
        ui.card(
            ui.card_header("Overview"),
            ui.output_image("image"), 
        ),
        ui.card(
            ui.card_header("Clusterview"), 
            ui.output_image("sub_image")
        )
    ),
    ui.layout_columns( 
        ui.card(
            ui.input_select("img", "Choose cluster:", choices=filenames)
        ),
        ui.card(
            ui.input_select("img2", "Choose cluster:", choices=filenames2)
        )
    ),
    title="EDA",
    fillable=True,
)  


def server(input, output, session):
    @render.image
    def image():
        x = input.img()
        dir = Path(__file__).resolve().parent
        img: ImgData = {"src": str(dir / f"results/images/{x}"), "width": "1000px"}
        return img

    @render.image
    def sub_image():
        x = input.img2()
        dir = Path(__file__).resolve().parent
        img: ImgData = {"src": str(dir / f"results/images/sub_images/{x}"), "width": "1000px"}
        return img

app = App(app_ui, server)

#shiny run app.py
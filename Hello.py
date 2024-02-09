# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import os
from streamlit.logger import get_logger

from ultralytics import YOLO


LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="NDU Classroom Navigation",
        page_icon="👋",
    )

    st.write("# Welcome to NDU Classroom Navigation")

    # st.sidebar.success("Select a demo above.")

    # st.markdown(
    #    """
    #    Streamlit is an open-source app framework built specifically for
    #    Machine Learning and Data Science projects.
    #    **👈 Select a demo from the sidebar** to see some examples
    #    of what Streamlit can do!
    #    ### Want to learn more?
    #    - Check out [streamlit.io](https://streamlit.io)
    #    - Jump into our [documentation](https://docs.streamlit.io)
    #    - Ask a question in our [community
    #      forums](https://discuss.streamlit.io)
    #    ### See more complex demos
    #    - Use a neural net to [analyze the Udacity Self-driving Car Image
    #      Dataset](https://github.com/streamlit/demo-self-driving)
    #    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    #"""
    #)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Create a unique filename to store the uploaded image
        filename = uploaded_file.name
        image_path = os.path.join("uploads", filename)
        os.makedirs("uploads", exist_ok=True)  # Create uploads folder if needed

        # Save the uploaded image
        with open(image_path, "wb") as f:
            f.write(uploaded_file.read())

        # Process the image
        processed_message = process_image(image_path)

        # Display the image and processing results
        st.image(image_path)
        st.write(processed_message)

def process_image(image_path):
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    results = model(image_path)  # predict on an image

    result_string = ""

    for box in results[0].boxes:
        result_string = result_string + "Class: {} \n".format(model.names[int(box.cls)])

    return result_string

if __name__ == "__main__":
    run()
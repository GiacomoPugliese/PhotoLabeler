# Streamlit App to use Amazon Rekognition API

import streamlit as st
import boto3
import os
from PIL import Image
import tempfile
from botocore.exceptions import ClientError
import logging
import zipfile
import io
from io import BytesIO
import base64
import shutil
import datetime
import re
from zipfile import ZipFile, ZIP_STORED
import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from datetime import datetime
import pyheif
import re

logging.basicConfig(level=logging.INFO)

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(
    page_title='SmartLabel',
    page_icon='camera'
)   
hide_streamlit_style = """ <style> #MainMenu {visibility: hidden;} footer {visibility: hidden;} </style> """ 
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Set AWS details (replace with your own details)
AWS_REGION_NAME = 'us-east-2'
AWS_ACCESS_KEY = 'AKIARK3QQWNWXGIGOFOH'
AWS_SECRET_KEY = 'ClAUaloRIp3ebj9atw07u/o3joULLY41ghDiDc2a'

# Initialize the S3 client
s3 = boto3.client('s3',
    region_name=AWS_REGION_NAME,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Initialize boto3 client for AWS Rekognition
client = boto3.client('rekognition',
    region_name=AWS_REGION_NAME,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

def save_file_locally(file, person_name):
    """
    Save the uploaded file to a local directory named after the person.

    :param file: File to save
    :param person_name: Name of the person. Used to create a directory for the person.
    :return: The path to the saved file
    """
    directory = os.path.join(os.getcwd(), collection_id, 'training_imgs', person_name)
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.read())
    return file_path


def resize_image(image, basewidth):
    img = Image.open(image)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    return img

def delete_collection(collection_id):
    try:
        response = client.delete_collection(CollectionId=collection_id)
        st.write('Deleted collection:', collection_id)
    except ClientError as e:
        logging.error(e)
        raise e
    
def create_collection(collection_id):
    try:
        response = client.create_collection(CollectionId=collection_id)
        pass
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceAlreadyExistsException':
            pass
        else:
            logging.error(e)
            raise e
        
def upload_file_to_s3(file, bucket, object_name):
    """
    Upload a file to an S3 bucket

    :param file: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    try:
        s3.upload_fileobj(file, bucket, object_name)
    except ClientError as e:
        logging.error('Error uploading file to S3: %s', e)
        return False
    logging.info('File uploaded successfully to bucket %s with key %s', bucket, object_name)
    return True


# Define function to add faces to the collection
def add_faces_to_collection(bucket, photo, collection_id, external_image_id):
    response = client.index_faces(
        CollectionId=collection_id,
        Image={'S3Object':{'Bucket':bucket,'Name':photo}},
        ExternalImageId=external_image_id,
        MaxFaces=5,
        QualityFilter="AUTO",
        DetectionAttributes=['ALL']
    )

    return response['FaceRecords']

def find_matching_faces(photo, collection_id):
    # Convert bytes to PIL Image
    image = Image.open(BytesIO(photo))

    # Attempt to detect faces in the photo
    faces = client.detect_faces(Image={'Bytes': photo})
    # print(f"Number of faces detected by detect_faces: {len(faces['FaceDetails'])}")
    
    # For each face, search for matches in the collection
    face_matches = []
    for i, face in enumerate(faces['FaceDetails']):
        bbox = face['BoundingBox']
        # Convert bbox to the format required by search_faces_by_image
        width, height = image.width, image.height
        left = int(bbox['Left']*width)
        top = int(bbox['Top']*height)
        face_width = int(bbox['Width']*width)
        face_height = int(bbox['Height']*height)

        # Add a buffer to the bounding box coordinates to make sure the entire face is included
        buffer = 20  # adjust this value as needed
        left = max(0, left - buffer)
        top = max(0, top - buffer)
        right = min(width, left + face_width + buffer)
        bottom = min(height, top + face_height + buffer)

        face_crop = image.crop((left, top, right, bottom))
        bytes_io = io.BytesIO()
        face_crop.save(bytes_io, format='JPEG')
        face_bytes = bytes_io.getvalue()

        try:
            response = client.search_faces_by_image(
                CollectionId=collection_id,
                Image={'Bytes': face_bytes},
                FaceMatchThreshold=70,
                MaxFaces=5
            )
            face_matches.extend([match['Face']['ExternalImageId'] for match in response['FaceMatches']])
            # print(f"Face {i+1}: matches found")
        except Exception as e:
            pass
            # print(f"Face {i+1}: error occurred - {str(e)}")

    # print(f"Number of face matches found: {len(face_matches)}")
    return face_matches

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{bin_file}">{file_label}</a>'
    return href

# Define function to list faces in the collection
def list_faces_in_collection(collection_id):
    response = client.list_faces(CollectionId=collection_id)
    return [face['ExternalImageId'] for face in response['Faces']]

# Define function to delete a face from the collection
def delete_face_from_collection(collection_id, face_id):
    response = client.delete_faces(CollectionId=collection_id, FaceIds=[face_id])
    return response['DeletedFaces']

# Define function to add a training image to an existing person
def add_training_image_to_person(collection_id, person_name, image):
    # Upload the image to S3 bucket
    object_name = f'{person_name}_{len(list_faces_in_collection(collection_id)) + 1}.jpg'
    upload_success = upload_file_to_s3(image, 'giacomo-aws-bucket', object_name)
    if upload_success:
        add_faces_to_collection('giacomo-aws-bucket', object_name, collection_id, object_name)

if 'person_names' not in st.session_state:
    st.session_state['person_names'] = []
    st.session_state['last_uploaded_file'] = None
    st.session_state['download_zip_created'] = False
    # delete_collection('your-collection-id')
    

st.title("Leadership Initiatives Photo Labeler")

with st.expander("Click to view full directions for this site"):
    st.subheader("Configure Training Data")
    st.write("- Either create or sign in to your program by entering a program ID (** a program ID is needed for all parts of the site **)")
    st.write("- Create student profiles and upload solo images of them to train the AI (** names must be in format 'FIRST_LAST' **)")
    st.write("- Delete student profile to clear their training data")
    st.subheader("Interns in System")
    st.write("- Displays all of the interns currently in your program's AI")
    st.subheader("Detect Interns in Photos")
    st.write("- Insert the folder link of your google drive containing your photos")
    st.write("- Click 'Start Processing' and allow the AI to sort the images into individual student folders directly into the drive (may take a while)")
    st.subheader("Renaming tool")
    st.write("- Upload a zip folder of images of program's students at a particular location")
    st.write("- Choose the custom file ending for that location (i.e. '_Jumpstart_Group_1' for a file you want named 'First_Last_2023_Jumpstart_Group_1')")
    st.write("- Click 'Start Renaming' and download a zip folder of the automatically renamed image files")

# Add a person or image, or delete a person
st.header('Configure Training Data')

collection_id = st.text_input("Enter your program ID", "")
if collection_id == '':
    collection_id = 'your-default-collection-id'
create_collection(collection_id)

# Create a directory named after the collection
os.makedirs(collection_id, exist_ok=True)

person_name = st.text_input("Enter the intern's name")
person_image = st.file_uploader('Upload a solo image of the intern', type=['jpg', 'png'])

if st.button('Add image'):
    if(collection_id == 'your-default-collection-id'):
        st.error("Please enter a collection id!")
    else:
        if person_name and person_image and collection_id:
            # Save the image locally
            file_path = save_file_locally(person_image, person_name)
            # Upload the image to S3 bucket
            with open(file_path, "rb") as f:
                upload_success = upload_file_to_s3(f, 'giacomo-aws-bucket', person_name)
            if upload_success:
                # Check if person already exists
                if person_name not in list_faces_in_collection(collection_id):
                    add_faces_to_collection('giacomo-aws-bucket', person_name, collection_id, person_name)
                    st.session_state['person_names'].append(person_name)
                    st.write('Person added successfully')
                else:
                    # If person already exists, just add the image to the person's existing images in the collection
                    st.write('Image added to existing person')
                # Clear the input fields
                person_name = None
                person_image = None
            else:
                st.write('Failed to upload image')
        else:
            st.write('Please enter a name, upload an image, and provide a collection ID')

if st.button('Delete intern'):
    if(collection_id == 'your-default-collection-id'):
        st.error("Please enter a collection id!")
    else:
        if person_name and collection_id:
            # Find the faceId of the person to delete
            face_id = next((face['FaceId'] for face in client.list_faces(CollectionId=collection_id)['Faces'] 
                            if face['ExternalImageId'] == person_name), None)
            if face_id:
                delete_face_from_collection(collection_id, face_id)
                st.write(f'Person {person_name} deleted successfully')
            else:
                st.write(f'Person {person_name} not found')
        else:
            st.write('Please enter a name and collection ID to delete')

# Display the list of person names
person_names = list_faces_in_collection(collection_id)
st.header(f'Interns in System ({len(person_names)})')
st.write(', '.join(person_names))
if len(person_names) == 0:
    'No interns added yet.'
st.button("Refresh page")

st.header('Detect Interns in Photos')
folder_link = st.text_input('Enter Google Drive Folder link')
start_processing = st.button('Start Processing')

if start_processing and folder_link:
    # Match any characters after the last slash in the URL
    match = re.search(r'\/([a-zA-Z0-9-_]+)$', folder_link)
    
    if(collection_id == 'your-default-collection-id' or match is None):
        if match is None:
            st.error('Invalid Google Drive link. Please make sure the link is correct.')
        else:
            st.error("Please enter a collection id!")
    else:
        # Build the service
        creds = service_account.Credentials.from_service_account_file('credentials.json')
        service = build('drive', 'v3', credentials=creds)

        # Request files in the folder
        results = service.files().list(q=f"'{folder_id}' in parents").execute()
        items = results.get('files', [])

        if not items:
            st.error("No files found.")
        else:
            total_files = len(items)

            person_images_dict = {}  # Dictionary to hold list of images for each person

            progress_report = st.empty()  # Create a placeholder for the progress report

            with st.spinner("Labeling images.."):
                group_photo_threshold = 15  # Number of people needed to categorize a photo as a group photo
                person_images_dict['Group Photos'] = []  # Initialize the 'Group Photos' category

                for i, file in enumerate(items, start=1):
                    try:
                        progress_report.text(f"Labeling progress: ({i}/{total_files})")  # Update the text in the placeholder
                        request = service.files().get_media(fileId=file['id'])

                        # Download and process the image
                        fh = io.BytesIO()
                        downloader = MediaIoBaseDownload(fh, request)
                        done = False
                        while done is False:
                            _, done = downloader.next_chunk()
                        
                        if file['name'].endswith('.heic') or file['name'].endswith('.HEIC'):
                            heif_file = pyheif.read(fh.getvalue())
                            img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode)
                            byte_arr = io.BytesIO()
                            img.save(byte_arr, format='JPEG')
                            byte_img = byte_arr.getvalue()
                        else:
                            img = resize_image(fh, 1000)  # Adjust the width as needed
                            byte_arr = io.BytesIO()
                            img.save(byte_arr, format='JPEG')  # Or format='PNG' if your images are PNG
                            byte_img = byte_arr.getvalue()

                        detected_persons = find_matching_faces(byte_img, collection_id)

                        # Check if the photo is a group photo
                        if len(set(detected_persons)) >= group_photo_threshold:
                            person_images_dict['Group Photos'].append(file['name'])
                            persons = ['Group Photos']
                        else:
                            persons = detected_persons

                        # Add the image to the list for each detected person
                        for person in set(persons):
                            if person not in person_images_dict:
                                person_images_dict[person] = []
                            person_images_dict[person].append(file['name'])

                            # Create or find the person-specific folder and copy the image into it
                            folder_query = f"name='{person}' and '{folder_id}' in parents"
                            folder_search = service.files().list(q=folder_query).execute().get('files', [])
                            if not folder_search:
                                # Create the folder if it does not exist
                                metadata = {'name': person, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [folder_id]}
                                folder = service.files().create(body=metadata, fields='id').execute()
                            else:
                                # Use the existing folder
                                folder = folder_search[0]

                            # Copy the file
                            current_year = datetime.now().year
                            new_file_name = f"{person}_{current_year}_{file['name']}"
                            copied_file = service.files().copy(fileId=file['id'], body={"name": new_file_name, "parents": [folder['id']]}).execute()

                    except Exception as e:
                        st.write(f"{file['name']} threw an error: {e}")
                        continue

            # Generate the text file
            with open(f'{collection_id}/labels.txt', 'w') as f:
                for person, images in person_images_dict.items():
                    f.write(f'{person}: {", ".join(images)}\n\n')

            st.session_state['download_zip_created'] = True  

# Replace the markdown link with a download button
if 'download_zip_created' in st.session_state and st.session_state['download_zip_created']:  

    with open(f'{collection_id}/labels.txt', 'r') as f:
        st.download_button(
            label="Download all textual labels",
            data=f.read(),
            file_name='labels.txt',
            mime='text/plain'
        )

st.header('Naming Tool')
folder_id_rename = st.text_input('Enter Google Drive Folder ID for Renaming')
file_name_ending = st.text_input('Enter your custom file name ending')
start_renaming = st.button('Start Renaming')

if start_renaming and folder_id_rename:
    if(collection_id == 'your-default-collection-id'):
        st.error("Please enter a collection id!")
    else:
        # Build the service
        creds = service_account.Credentials.from_service_account_file('credentials.json')
        service = build('drive', 'v3', credentials=creds)

        # Request files in the folder
        results = service.files().list(q=f"'{folder_id_rename}' in parents").execute()
        items = results.get('files', [])

        if not items:
            st.error("No files found.")
        else:
            total_files = len(items)

            progress_report = st.empty()  # Create a placeholder for the progress report

            for i, file in enumerate(items, start=1):
                try:
                    progress_report.text(f"Renaming progress: ({i}/{total_files})")  # Update the text in the placeholder

                    # Extract file extension
                    file_ext = os.path.splitext(file['name'])[1]

                    if '_2023_' in file['name']:
                        # Remove all characters past '_2023_' in the file name
                        new_file_name = re.sub(r'_2023_.*', '', file['name'])

                        # Append the custom file name ending and the file extension
                        new_file_name += f'_2023_{file_name_ending}{file_ext}'

                        # Rename the file
                        service.files().update(fileId=file['id'], body={"name": new_file_name}).execute()

                except Exception as e:
                    st.write(f"Error renaming {file['name']}: {e}")
                    continue

            st.success("All files renamed successfully!")


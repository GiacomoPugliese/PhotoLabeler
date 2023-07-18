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
import threading
from multiprocessing import Pool
from googleapiclient import errors
import time
from google_auth_oauthlib.flow import InstalledAppFlow
import requests
import json
from google.oauth2.credentials import Credentials
import webbrowser
# import pyheif

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
    st.session_state['creds'] = None
    # st.session_state['auth'] = False
    # delete_collection('your-collection-id')
    

st.title("Leadership Initiatives Photo Labeler")

with st.expander("Click to view full directions for this site"):
    st.subheader("Configure Training Data")
    st.write("- Either create or sign in to your program by entering a program ID (** a program ID is needed for all parts of the site **)")
    st.write("- Create student profiles and upload solo images of them to train the AI (** names must be in format FIRST_LAST **)")
    st.write("- Delete student profile to clear their training data if needed")
    st.subheader("Interns in System")
    st.write("- Displays all of the interns currently in your program's AI")
    st.subheader("Detect Interns in Photos")
    st.write("- Insert the folder link of your google drive containing your photos")
    st.write("- Share the folder with giacomo@photolabeler-393105.iam.gserviceaccount.com")
    st.write("- Click 'Start Processing' and allow the AI to sort the images into individual student folders directly into the drive (may take a while)")
    st.subheader("Renaming tool")
    st.write("- Insert the folder links of your google drive containing program's students at a particular location")
    st.write("- Choose the custom file ending for that location (i.e. ending would be '_Jumpstart_Group_1' for a file you want named 'Joe_Random_2023_Jumpstart_Group_1')")
    st.write("- Click 'Start Renaming' and download a zip folder of the automatically renamed image files")


st.header('User Credentials')
collection_id = st.text_input("Enter your program ID", "")
if collection_id == '':
    collection_id = 'your-default-collection-id'
collection_id = 'your-collection-id'
create_collection(collection_id)

with st.expander("Authenticate Google Account"):

    # Request OAuth URL from the FastAPI backend
    response = requests.get(f"{'https://photo-labeler-842ac8d73e7a.herokuapp.com'}/auth?user_id={collection_id}")
    if response.status_code == 200:
        # Get the authorization URL from the response
        auth_url = response.json().get('authorization_url')
        
        # Redirect user to the OAuth URL
        # webbrowser.open(auth_url, new=2)
        st.markdown(f"[Click to begin authentication process]({auth_url})")
        # st.session_state['auth'] = True

if True:    
    if st.button("Finalize Google Authentication"):
        # Request token from the FastAPI backend
        response = requests.get(f"{'https://photo-labeler-842ac8d73e7a.herokuapp.com'}/token/{collection_id}")
        if response.status_code == 200:
            st.session_state['creds'] = response.json().get('creds')
            print(st.session_state['creds'])
            st.success("Google account successfully authenticated.")
        else:
            st.error('Failed to retrieve credentials')


# Add a person or image, or delete a person
st.header('Configure Training Data')


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

########################################################################################
#    DETECT SECTION

def make_request_with_exponential_backoff(request):
    for n in range(0, 5):
        try:
            return request.execute()
        except Exception as e:
            if isinstance(e, errors.HttpError) and e.resp.status == 403:
                time.sleep((2 ** n) + random.random())
            else:
                raise e
    print("Request failed after 5 retries")
    return None


def process_file(file, service, folder_id, person_images_dict, group_photo_threshold, collection_id, person_folder_dict):
    st.write(f"{file['name']} started")
    try:
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
            img = resize_image(fh, 1000)
            byte_arr = io.BytesIO()
            img.save(byte_arr, format='JPEG')
            byte_img = byte_arr.getvalue()

        detected_persons = find_matching_faces(byte_img, collection_id)

        if len(set(detected_persons)) >= group_photo_threshold:
            person_images_dict['Group Photos'].append(file['name'])
            persons = ['Group Photos']
        else:
            persons = detected_persons

        for person in set(persons):
            if person not in person_images_dict:
                person_images_dict[person] = []
            person_images_dict[person].append(file['name'])

            # Get the person's folder from the dictionary
            folder = person_folder_dict[person]

            current_year = datetime.now().year
            new_file_name = f"{person}_{current_year}_{file['name']}"
            copied_file = make_request_with_exponential_backoff(service.files().copy(fileId=file['id'], body={"name": new_file_name, "parents": [folder['id']]}))


    except Exception as e:
        st.write(f"{file['name']} threw an error: {e}")

    print(file['name'] + " labeled")

def process_file_wrapper(args):
    return process_file(*args)

st.header('Detect Interns in Photos')
folder_link = st.text_input('Enter Google Drive Folder link')
start_processing = st.button('Start Processing')

    
if start_processing:
    if not folder_link:
        st.error("Please upload your google drive folder")
    # elif not st.session_state['auth']:
    #     st.error("Please authenticate google account")
    else:
        match = re.search(r'\/([a-zA-Z0-9-_]+)$', folder_link)
        if(collection_id == 'your-default-collection-id' or match is None):
            if match is None:
                st.error('Invalid Google Drive link. Please make sure the link is correct.')
            else:
                st.error("Please enter a collection id!")
        else:
            folder_id = match.group(1)
            
            CLIENT_SECRET_FILE = 'credentials.json'
            API_NAME = 'drive'
            API_VERSION = 'v3'
            SCOPES = ['https://www.googleapis.com/auth/drive']

            # flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            # creds = flow.run_local_server(port=8005)

            # Load client secrets from your credential file

            with open(CLIENT_SECRET_FILE, 'r') as f:
                client_info = json.load(f)['web']

            creds_dict = st.session_state['creds']
            creds_dict['client_id'] = client_info['client_id']
            creds_dict['client_secret'] = client_info['client_secret']
            creds_dict['refresh_token'] = creds_dict.get('_refresh_token')
            try:
                # Create Credentials from creds_dict
                creds = Credentials.from_authorized_user_info(creds_dict)

                # Call the Drive v3 API
                service = build(API_NAME, API_VERSION, credentials=creds)
            except:
                st.error("Please refresh the page and retry Google authentication.")

            # Create a dictionary to store each person's folder
            with st.spinner("Creating folders"):
                person_folder_dict = {}
                for person in person_names:
                    folder_query = f"name='{person}' and '{folder_id}' in parents and trashed=false"
                    folder_search = make_request_with_exponential_backoff(service.files().list(q=folder_query))
                    if not folder_search.get('files', []):
                        metadata = {'name': person, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [folder_id]}
                        folder = make_request_with_exponential_backoff(service.files().create(body=metadata, fields='id'))
                    else:
                        print("folder detected!")
                        print(folder_search)  # print the full response
                        folder = folder_search.get('files', [])[0]

                    person_folder_dict[person] = folder

                person_images_dict = {}
                person_images_dict['Group Photos'] = []
                group_photo_threshold = 15

            progress_report = st.empty()

            with st.spinner("Labeling images.."):
                total_files = 0  # initialize total file counter
                labeled_files = 0  # initialize labeled file counter
                page_token = None

                #retreive total amount of files
                response = make_request_with_exponential_backoff(service.files().list(q=f"'{folder_id}' in parents and trashed=false",
                                                                                        spaces='drive',
                                                                                        fields='nextPageToken, files(id, name)',
                                                                                        pageToken=page_token,
                                                                                        pageSize=1000))
                items = response.get('files', [])
                arguments = [(file, service, folder_id, person_images_dict, group_photo_threshold, collection_id, person_folder_dict,) for file in items]
                total_files = len(items) - len(person_names)
                progress_report.text(f"Labeling progress: ({0}/{total_files})")

                #make a request to get rid of the folders first
                response = make_request_with_exponential_backoff(service.files().list(q=f"'{folder_id}' in parents  and trashed=false", spaces='drive', fields='nextPageToken, files(id, name)', pageToken=page_token,pageSize=len(person_names)))
                items = response.get('files', [])
                page_token = response.get('nextPageToken', None)

                while True:
                    response = make_request_with_exponential_backoff(service.files().list(q=f"'{folder_id}' in parents and trashed=false",
                                                                                        spaces='drive', 
                                                                                        fields='nextPageToken, files(id, name)',
                                                                                        pageToken=page_token,
                                                                                        pageSize=10))
                    items = response.get('files', [])
                    arguments = [(file, service, folder_id, person_images_dict, group_photo_threshold, collection_id, person_folder_dict,) for file in items]

                    with Pool(processes=10) as pool:
                        pool.map(process_file_wrapper, arguments)

                    labeled_files += len(items)
                    progress_report.text(f"Labeling progress: ({labeled_files}/{total_files})")  # update progress bar with the ratio of labeled files to total files

                    page_token = response.get('nextPageToken', None)
                    if page_token is None:
                        break

                with open(f'{collection_id}/labels.txt', 'w') as f:
                    for person, images in person_images_dict.items():
                        f.write(f'{person}: {", ".join(images)}\n\n')

                st.session_state['download_zip_created'] = True  

if 'download_zip_created' in st.session_state and st.session_state['download_zip_created']:  

    with open(f'{collection_id}/labels.txt', 'r') as f:
        st.download_button(
            label="Download all textual labels",
            data=f.read(),
            file_name='labels.txt',
            mime='text/plain'
        )

##############################################################################################

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


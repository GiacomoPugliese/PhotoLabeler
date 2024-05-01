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
import random
from PIL import ImageOps, ExifTags
import uuid
import glob
from streamlit_javascript import st_javascript
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from googleapiclient.http import MediaFileUpload
import traceback
from urllib.parse import urlparse, parse_qs
import pyheif

logging.basicConfig(level=logging.INFO)

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

def convert_heic_to_jpeg(file_name):
    # Full path to the original HEIC file
    original_file_path = os.path.abspath(file_name)

    # Read HEIC file
    heif_file = pyheif.read(original_file_path)
    
    # Convert to PIL Image
    image = Image.frombytes(
        heif_file.mode, 
        heif_file.size, 
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )

    # Save as JPEG
    jpeg_filename = f'{uuid.uuid4()}.jpg' 
    image.save(jpeg_filename, 'JPEG')

    # Read the saved jpeg file and convert it to bytes
    with open(jpeg_filename, 'rb') as img_file:
        byte_img = img_file.read()

    # Delete both the original .heic file and converted .jpg file
    os.remove(original_file_path)
    os.remove(jpeg_filename)

    return byte_img

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

# Define function to list faces in the collection
def list_faces_in_collection(collection_id):
    response = client.list_faces(CollectionId=collection_id)
    return [face['ExternalImageId'] for face in response['Faces']]

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
    try:
        return request.execute()
    except:
        return None
    
def sanitize_name(name):
    """Sanitize names to match AWS requirements for ExternalImageId"""
    # Remove everything after a hyphen, an underscore or ".jpg"
    name = re.sub(r' -.*|_.*|\.jpg|\.JPG|\.jpeg|\.JPEG|\.png|\.PNG|\.heic|\.HEIC', '', name)
    # Keep only alphabets, spaces, and hyphens
    name = re.sub(r'[^a-zA-Z \-]', '', name)
    # Replace hyphens and spaces with underscores
    name = re.sub(r'[- ]', '_', name)
    # Replace multiple underscores with a single underscore
    name = re.sub(r'[_]+', '_', name)
    # Remove leading underscore if it exists
    if name.startswith('_'):
        name = name[1:]
    return name

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

def resize_image(image, basewidth):
    img = Image.open(image)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    return img

def process_folder(folder, service, collection_id, parent_folder):
    has_training_image = False
    temp_file_path = None
    unique_filename = None

    query = f"'{folder['id']}' in parents and mimeType != 'application/vnd.google-apps.folder' and trashed = false"
    intern_images = service.files().list(q=query).execute().get('files', [])

    query = f"'{parent_folder}' in parents and name='Training Images' and trashed = false"
    results = service.files().list(q=query).execute().get('files', [])
    
    if results:
        training_images_folder_id = results[0]['id']
    else:
        file_metadata = {
            'name': 'Training Images',
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_folder]
        }
        training_images_folder = service.files().create(body=file_metadata, fields='id').execute()
        training_images_folder_id = training_images_folder['id']

    for img in intern_images:
        try:
            if 'bio' in img['name'].lower() or 'headshot' in img['name'].lower():
                image_id = img['id']
                image_name = img['name']
                request = service.files().get_media(fileId=image_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    _, done = downloader.next_chunk()
 
                    
                if image_name.endswith('.heic') or image_name.endswith('.HEIC'):
                    # Save the HEIC file to the local directory first
                    unique_heic_filename = f'{uuid.uuid4()}.heic'
                    fh = io.BytesIO()
                    request = service.files().get_media(fileId=image_id)
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while done is False:
                        _, done = downloader.next_chunk()

                    # Now fh holds the file content
                    with open(unique_heic_filename, 'wb') as f:  # Open a file in binary mode for writing
                        f.write(fh.getvalue())  # Write the content

                    
                    # Then convert the saved HEIC file to JPEG
                    byte_img = convert_heic_to_jpeg(unique_heic_filename)

                else:
                    img_io = io.BytesIO(fh.getvalue())
                    img = resize_image(img_io, 1000)
                    if img.mode != 'RGB':  # Convert to RGB if not already
                       img = img.convert('RGB')
                    byte_arr = io.BytesIO()
                    img.save(byte_arr, format='JPEG')
                    byte_img = byte_arr.getvalue()

                intern_name = folder['name']
                print(intern_name)
                sanitized_intern_name = sanitize_name(intern_name)
                print(sanitized_intern_name)
                if sanitized_intern_name not in list_faces_in_collection(collection_id):
                    upload_success = upload_file_to_s3(io.BytesIO(byte_img), 'leadership-aws-bucket', sanitized_intern_name)
                    if upload_success:
                        print(add_faces_to_collection('leadership-aws-bucket', sanitized_intern_name, collection_id, sanitized_intern_name))
                        print(f'Person {sanitized_intern_name} added successfully')
                        # Copy the original image to 'Training Images' folder in Google Drive
                        file_extension = os.path.splitext(image_name)[1]  # Extracting the file extension from the original name
                        file_metadata = {
                            'name': f'{sanitized_intern_name}{file_extension}',  # Using sanitized name and original extension
                            'parents': [training_images_folder_id]
                        }
                        copied_file = service.files().copy(fileId=image_id, body=file_metadata).execute()
                    else:
                        print('Failed to upload image')
                else:
                    print(f'{sanitized_intern_name} already in system')

                # After processing the image and saving to byte_img:

                
                has_training_image = True
                break 

        except Exception as e:
            print(f"{image_name} threw an error: {e}")
            traceback.print_exc()  # This line prints the full traceback

    if not has_training_image:
        print(folder['name'] + 'has no training data!')
        intern_name = folder['name']
        # split the intern_name on ' - ', and return the part before ' - '
        error_name = intern_name.split(' - ', 1)[0] if ' - ' in intern_name else intern_name
        return error_name  # return the error_name in case of error

    return None  # return None if there was no error

def process_file(file, service, folder_id, person_images_dict, group_photo_threshold, collection_id, person_folder_dict):
    print(f"{file['name']} started")
    try:
        # Initialize persons
        persons = []
        request = service.files().get_media(fileId=file['id'])
        # Download and process the image
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            _, done = downloader.next_chunk()

        if file['name'].endswith('.heic') or file['name'].endswith('.HEIC'):
            try:
                heif_file = pyheif.read(fh.getvalue())
                img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode)
                byte_arr = io.BytesIO()
                img.save(byte_arr, format='JPEG')
                img.save('converted3.jpg')
                byte_img = byte_arr.getvalue()
            except:
                img_io = io.BytesIO(fh.getvalue())
                img = resize_image(img_io, 1000)
                if img.mode != 'RGB':  # Convert to RGB if not already
                    img = img.convert('RGB')
                byte_arr = io.BytesIO()
                img.save(byte_arr, format='JPEG')
                byte_img = byte_arr.getvalue()
                
        else:  # This will cover both .jpg and .png files
            img_io = io.BytesIO(fh.getvalue())
            img = resize_image(img_io, 1000)
            if img.mode != 'RGB':  # Convert to RGB if not already
                img = img.convert('RGB')
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
            # Check if file already exists in the destination folder
            search_response = make_request_with_exponential_backoff(service.files().list(q=f"name='{new_file_name}' and '{folder['id']}' in parents and trashed=false",
                                                                                        spaces='drive',
                                                                                        fields='files(id, name)'))

            # If file does not exist, then copy it
            if not search_response.get('files', []):
                copied_file = make_request_with_exponential_backoff(service.files().copy(fileId=file['id'], body={"name": new_file_name, "parents": [folder['id']]}))


    except Exception as e:
        print(f"{file['name']} threw an error: {e}")
        traceback.print_exc()  # This line prints the full traceback

    # Generate a unique filename using uuid library
    unique_filename = str(uuid.uuid4()) + '.txt'
    with open(f'{collection_id}/labels/{unique_filename}', 'w') as f:
        # Write the image name and persons detected to the file
        f.write(f"{file['name']}: {', '.join(set(persons))}")

    print(f"{file['name']}: {', '.join(set(persons))}")
    return file['name']

def create_folder_wrapper(arg):
    service, destination_folder_id, person = arg
    folder_query = f"name='{person}' and '{destination_folder_id}' in parents and trashed=false"
    folder_search = make_request_with_exponential_backoff(service.files().list(q=folder_query))
    if not folder_search.get('files', []):
        metadata = {'name': person, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [destination_folder_id]}
        folder = make_request_with_exponential_backoff(service.files().create(body=metadata, fields='id'))
    else:
        folder = folder_search.get('files', [])[0]
    return person, folder

def process_file_wrapper(args):
    return process_file(*args)


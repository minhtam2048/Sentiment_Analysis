from django.http.response import JsonResponse

from rest_framework.parsers import JSONParser
from rest_framework import status
from rest_framework.decorators import api_view

from posts.models import Post
from posts.serializers import PostSerializer

from posts.scripts.util import read_file, tokenize, make_embedding, text_to_sequences
from posts.scripts.constant import DEFAULT_MAX_FEATURES
from posts.scripts.rnn import SARNNKerasCPU
from posts.scripts.cnn import TextCNN,VDCNN

import os
import numpy as np
import pandas as pd

from numpy import save,load
import pickle
import json
import keras.backend.tensorflow_backend as tb

# embedding_path = '/posts/baomoi.model.bin'
embedding_path = 'E:\Sentiment-Analysis\HTDPT\posts\baomoi.model.bin'
annoy_path = 'E:\Sentiment-Analysis\HTDPT\posts\embeddings\annoy.index'

max_features = DEFAULT_MAX_FEATURES
should_find_threshold = False
should_mix = False
return_prob = True
trainable = True 
use_additive_emb = False 
augment_size = 7650 
use_sim_dict = True 
print_model = True 

model_dict = {
#     'RNNKeras': RNNKeras,
#     'RNNKerasCPU': RNNKerasCPU,
#     'LSTMKeras': LSTMKeras,
    'SARNNKerasCPU': SARNNKerasCPU,
#     'SARNNKeras': SARNNKeras,
    'TextCNN': TextCNN,
#     'LSTMCNN': LSTMCNN,
    'VDCNN': VDCNN
}



def model_predict_comment(model,model_path,comment):
    resultInNumber = 0.0
    resultInBoolean = False

    tb._SYMBOLIC_SCOPE.value = True

    #load parameter
    embedding_mat = load(model_path + '/embedding_mat.npy')
    with open(model_path + '/embed_size.dat', "rb") as f:
        embed_size = pickle.load(f)
    with open(model_path + '/OPTIMAL_THRESHOLD.dat', "rb") as f:
        OPTIMAL_THRESHOLD = pickle.load(f)
    word_map = json.load(open(model_path + '/word_map.csv'))
    model = model(
        embeddingMatrix=embedding_mat,
        embed_size=embed_size,
        max_features=embedding_mat.shape[0],
        trainable = True,
        use_additive_emb = use_additive_emb
    )
    model.load_weights('{}/models.hdf5'.format(model_path))
    if len(comment) !=0 :
        comment_tokenizes_texts = tokenize(comment)
        comment_id_texts = text_to_sequences(comment_tokenizes_texts, word_map,checkmap=True)
        comment_prediction = model.predict(comment_id_texts)
        # print('Comment Prediction : ',(comment_prediction > OPTIMAL_THRESHOLD).astype(np.float),2)
        trustNumber = (comment_prediction[0][0]).astype(float)
        trustNumber = round(trustNumber, 2)
        if trustNumber < 0.5:
            trustNumber = 1.0 - trustNumber
            trustNumber = round(trustNumber, 2)

        if (comment_prediction > OPTIMAL_THRESHOLD).astype(np.int8) == 1:
            resultBoolean = False
        else:
            resultBoolean = True
    return trustNumber, resultBoolean

# model_path = '/posts/SARNNKerasCPU 2020-05-05-09_05_15-version'

model_path = 'E:\Sentiment-Analysis\HTDPT\posts\SARNNKerasCPU 2020-05-05-09_05_15-version'


@api_view(['GET', 'POST', 'PUT', 'DELETE'])
def post_list(request):
    if request.method == 'GET':
        posts = Post.objects.all()

        content = request.GET.get('content', None)
        if content is not None:
            posts = posts.filter(content__icontains=content)

        posts_serializer = PostSerializer(posts, many=True)
        return JsonResponse(posts_serializer.data, safe=False)

    elif request.method == 'POST':
        comment_list = []
        post_data = JSONParser().parse(request)
        post_serializer = PostSerializer(data=post_data, many=True)

        if post_serializer.is_valid():
            # print(post_serializer.data)
            
            for i in post_serializer.data:
                str = []
                # print(i.get('content'))
                str.append('*' + i.get('content') + '*')
                # comment_list.append('*' + i.get('content') + '*')
                trustNumber, resultBoolean = model_predict_comment (model_dict['SARNNKerasCPU'],model_path, str)
                print(trustNumber)
                print(resultBoolean)
                i.update(resultInNumber=trustNumber)
                i.update(resultInBoolean=resultBoolean)

            # model_predict_comment(model_dict['SARNNKerasCPU'],model_path,comment_list)
            # post_serializer.save()
            return JsonResponse(post_serializer.data, status=status.HTTP_201_CREATED, safe=False)
        return JsonResponse(post_serializer.data, status=status.HTTP_400_BAD_REQUEST)
    
    elif request.method == 'PUT':
        post_data = JSONParser().parse(request)
        post_serializer = PostSerializer(data=post_data, many=True)
        print(post_serializer)
        if post_serializer.is_valid():
            post_serializer.save()
            return JsonResponse(post_serializer.data, safe=False)
        return JsonResponse(post_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
 

    elif request.method == 'DELETE':
        count = Post.objects.all().delete()
        return JsonResponse({'message': '{} posts were deleted successfully'.format(count[0])}, status=status.HTTP_204_NO_CONTENT)


# @api_view(['GET', 'PUT', 'DELETE'])
# def post_detail(request, pk):
#     try:
#         post = Post.objects.get(pk=pk)
#     except Post.DoesNotExist:
#         return JsonResponse({'message': 'The post does not exist'}, status=status.HTTP_404_NOT_FOUND)

#     if request.method == 'GET':
#         post_serializer = PostSerializer(post)
#         return JsonResponse(post_serializer.data)

#     elif request.method == 'PUT':
#         post_data = JSONParser().parse(request)
#         post_serializer = PostSerializer(post, data=post_data)
#         if post_serializer.is_valid():
#             post_serializer.save()
#             return JsonResponse(post_serializer.data)
#         return JsonResponse(post_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#     elif request.method == 'DELETE':
#         post.delete()
#         return JsonResponse({'message': 'Post was deleted'}, status=status.HTTP_204_NO_CONTENT)


# @api_view(['GET'])
# def post_list_result(request):
#     posts = Post.objects.all()
#     if request.method == 'GET':
#         posts_serializer = PostSerializer(posts, many=True)
#         return JsonResponse(posts_serializer.data, safe=False)

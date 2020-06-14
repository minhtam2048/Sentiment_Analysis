from rest_framework import serializers
from rest_framework.serializers import RelatedField
from posts.models import Post, PostLanguage


class PostSerializer(serializers.ModelSerializer):

    class Meta:
        model = Post
        fields = ('id', 'content', 'resultInNumber', 'resultInBoolean')


class PostRelatedField(RelatedField):
    def to_representation(self, obj):
        data = {
            'id': obj.id,
            'content': obj.content,
            'resultInNumber': obj.resultInNumber,
            'resultInBoolean': obj.resultInBoolean
        }
        return data

    def to_internal_value(self, pk):
        return Post.objects.get(id=pk)


class PostLanguageSerializer(serializers.ModelSerializer):
    posts = PostSerializer(many=True)
    class Meta:
        model = PostLanguage
        fields = ('language', 'posts')

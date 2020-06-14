from django.db import models

# Create your models here.
class Post(models.Model):
    content = models.TextField(blank=False)
    resultInNumber = models.FloatField(default=0.0)
    resultInBoolean = models.BooleanField(default=False)


class PostLanguage(models.Model):
    language = models.TextField(blank=False)
    posts = models.ForeignKey(Post, default='', on_delete=models.CASCADE)


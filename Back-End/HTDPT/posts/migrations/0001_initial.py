# Generated by Django 3.0.7 on 2020-06-11 04:11

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Post',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('content', models.TextField()),
                ('resultInNumber', models.FloatField(default=0.0)),
                ('resultInBoolean', models.BooleanField(default=False)),
            ],
        ),
    ]

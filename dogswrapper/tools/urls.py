"""Lemix URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.9/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Add an import:  from blog import urls as blog_urls
    2. Import the include() function: from django.conf.urls import url, include
    3. Add a URL to urlpatterns:  url(r'^blog/', include(blog_urls))
"""
from django.conf.urls import url, include
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
import Inventory.views as inv

urlpatterns = [

    url(r'^admin/', admin.site.urls),

    # ex: /Inventory/
    url(r'^$', inv.index, name='index'),

    url(r'^inventory/', include('Inventory.urls')),

    url(r'^cinvoicer/', include('CInvoicer.urls')),

    url(r'^invimport/', include('inv_import.urls')),

    url(r'^purchasemanager/', include('PurchaseManager.urls')),

    url(r'^bulkimport/', include('BulkImport.urls')),

    url(r'^updatesite/', include('UpdateSite.urls')),

    url(r'^pdupdater/', include('PDUpdater.urls')),

    url(r'^taggen/', include('TagGen.urls')),

    url(r'^logout_user/', inv.logout_user, name='logout_user'),

    # ex: /login/
    url(r'^login/', inv.loginpage, name='loginpage'),

    # ex: /loginck/
    url(r'^loginck/', inv.loginck, name='loginck')

] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
'''
url(
            r'^media/(?P<path>.*)$',
            'django.views.static.serve',
            {'document_root': '/home/ExtremePro_Workspace/AppsExtremeProjectOutdoor/Lemix/', 'show_indexes': True }
    ),
'''
'''
urlpatterns +=
urlpatterns +=  url(r'^static/(?P<path>.*)$', 'django.views.static.serve', {'document_root': '/home/teo/MYWORKSPACE/dev/Lemix/', })
'''

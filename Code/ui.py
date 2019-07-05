#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os,cv2
import PIL.Image, PIL.ImageTk
import numpy as np
from tkinter import *
from tkinter.font import Font,BOLD
from tkinter.ttk import Style,Treeview
import numpy as np
import cv2
from math import sqrt
import copy
import scipy.spatial
import math
from tkinter.messagebox import *
from win32api import GetSystemMetrics
from shutil import copyfile
from tkinter.filedialog import askopenfilename,askdirectory

import segmentation
import extraction_classification

class Interface(Frame):
    
    
    def __init__(self, fenetre, **kwargs):
        
        Frame.__init__(self, fenetre, width=GetSystemMetrics(0), height=GetSystemMetrics(1), relief='ridge', **kwargs)    #création d'une interface qui hérite de la classe Frame de tkinter
        self.pack(fill=BOTH)

        self.canva=Canvas(self,width=0.86*self.winfo_screenwidth(),height=0.911*self.winfo_screenheight())   #création d'une zone de dessin où afficher l'image
        self.canva.grid(row=0,column=0,columnspan=2000,rowspan=500)
        
        bookman20=Font(family='Bookman', size=13, weight=BOLD) #police personnalisée pour les boutons

        self.bouton_ouvrir = Button(self, text="Ouvrir", bg="white smoke", height=1, command=self.charger_image, font=bookman20)  #création du bouton ouvrir (une image)        
        self.bouton_aide=Button(self, text="Aide", bg="white smoke", height=1,command=self.aide, font=bookman20) #création du bouton d'aide 
        self.bouton_aide.grid(row=0, column=4)
        self.bouton_ouvrir.grid(row=0,column=0)

        self.bouton_segmentation = Button(self, text="Segmentation",fg="black", height=1 , bg="OliveDrab2", command=self.segmenter, font=bookman20)#création du bouton segmentation   
        self.bouton_segmentation.grid(row=0, column=1)
        
        self.bouton_extraction = Button(self, text="Extraction & Classification",fg="black" , height=1,bg="coral1", command=self.extraire_caracs, font=bookman20)#création du bouton extraction&classification  
        self.bouton_extraction.grid(row=0,column=2)
    
        self.bouton_sauvegarder = Button(self, text="Sauvegarde",bg="white smoke", height=1, fg='black', command=self.sauvegarder, font=bookman20)#création du bouton de sauvegarde 
        self.bouton_sauvegarder.grid(row=0,column=3)
     
    #appuyer sur le bouton aide produit une fenêtre d'information   
    def aide(self):
        showinfo(title="Aide", message="Le mélanome est le cancer de la peau le plus mortel. D'après l'OMS, il est la cause d'environ 60000 décès par an dans le monde. Le dépistage précoce du mélanome est important pour traiter la maladie au plus vite et retirer la tumeur par une simple excision.\n\nOuvrir : Permet de charger et d'afficher une image dont la taille convient à celle de l'écran.\n\nSegmentation : A partir de l'image ouverte, un masque binaire qui délimite le grain de beauté est créé. Pour se faire, on se restreint d'abord au canal bleu de l'image. On supprime ensuite les poils apparents. La détection du grain de beauté se fait par seuillage via la méthode d'Otsu. Enfin, les composantes résiduelles sont retirées.\n\nExtraction&Classification : L’extraction de features consiste à déterminer différentes caractéristiques des grains de beauté étudiés grâce à des méthodes codées en Python. Ce processus suit la méthode ABCD (asymétrie, bordure, couleur et diamètre).\nLes données obtenues sont par la suite utilisées dans des algorithmes de Machine Learning préalablement entraînés pour déclarer si le grain de beauté en question est bénin ou malin.", default=OK, icon=INFO)
    #la fonction qui s'execute lors de l'appui sur le bouton "Ouvrir"
    def charger_image(self):
        global im, im_seg,results              
        im,im_seg,results=None,None,None        #cette ligne et la suivante permette de réinitialiser les variables images, image_segmentée et les résultats à l'ouverture d'une image
        del im,im_seg,results                     
        self.canva.delete("all")
        global filepath   #on déclare filepath global pour qu'elle soit accessible dans les autres fonctions de boutons
        filepath = askopenfilename(filetypes=[("Image Files","*.jpg;*.png")])
        im=cv2.imread(filepath)
        im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  #opencv ouvre des image en BGR. On la convertit en RGB
        he,wi=im.shape[:2]
        global wsize,baseheight
        baseheight = self.winfo_height()
        hpercent = (baseheight / float(he))
        wsize = int((float(wi) * float(hpercent)))
        im_redimensionnee = cv2.resize(im,(wsize, baseheight)) #on redimensionne l'image de sorte que sa hauteur soit celle de la fenêtre principale
        #self.canva['width']= wsize+ (self.winfo_screenwidth()-wsize)/4.25
        global photo
        photo= PIL.ImageTk.PhotoImage(PIL.Image.fromarray(im_redimensionnee))
        fenetre.geometry("%dx%d+0+0" % (wsize, baseheight)) #on redimensionne la fenêtre à la taille de l'image
        self.canva.create_image(0,0,image =photo, anchor='nw') #on affiche l'image sur le canva
        
    #la fonction qui s'execute lors de l'appui sur le bouton "Segmentation"
    def segmenter(self):
        
        try:
            global im_seg #on déclare im_seg global pour qu'elle soit accessible dans les autres fonctions de boutons
            im_seg=segmentation.main(filepath) #on execute la fonction main de la semgentation qui retourne le masque binaire
            #im_seg=im
            im_seg=im_seg.astype('uint8')
            im1, detectedContours, hierarchy = cv2.findContours(im_seg,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #à partir du masque binaire, on dessine les contouts du grain de beauté sur l'image initiale
            for contour in detectedContours :
                cv2.drawContours(im,contour, -1, (0,0,255), 3)
            im_redimensionnee = cv2.resize(im,(wsize, baseheight))
            self.canva['width']= wsize+ (self.winfo_screenwidth()-wsize)/4.25
            global photo
            photo= PIL.ImageTk.PhotoImage(PIL.Image.fromarray(im_redimensionnee))
            self.canva.delete("all")
            self.canva.create_image(0,0,image =photo, anchor='nw') #on affiche l'image segmentée redimensionnée
            
        except NameError:
            im_exists = 'filepath' in locals() or 'filepath' in globals()
            if not im_exists:   
                showerror(title="Erreur", message="Vous n'avez pas ouvert d'image.", default=OK, icon=ERROR) #ceci est un message de précaution si aucune image n'a encore été ouverte et qu'on appuie sur le bouton de segmentation


    #la fonction qui s'execute lors de l'appui sur le bouton "Extraction&Classification"
    def extraire_caracs(self):
        
        try:
            m=im     #Si im ou im_seg n'est pas définie (on n'a pas encore ouvert d'image ou pas encore segmentée l'image ouverte, on aura un NameError d'où un message de précaution de notre part
            n=im_seg
            scores = Tk()           #on va afficher une nouvelle fenêtre qui contiendra les résultats.
            scores.title('Extraction des caractéristiques')

            style = Style(scores)

            style.configure('Treeview', rowheight=40)                 #Permet de changer la taille du tableau qu'on va créer, ainsi que la police d'écriture
            style.configure('Treeview', font=(None, 15))
            style.configure('Treeview.Heading', font=(None, 18, 'bold'))                
            cols = ('Caractéristiques', 'Scores')
            listBox = Treeview(scores, height=7, columns=cols, show='headings') #Le tableau est un Treeview
            for col in cols:
                listBox.heading(col, text=col)       #on regle les titres des colonnes du tableau et leur taille
                listBox.column(col, width=int(1/6*GetSystemMetrics(0)))

            listBox.grid(row=0, column=0, columnspan=2)

            L=extraction_classification.ABCDbis(filepath,255*im_seg)   #les  résultats de l'extraction des caractéristiques
            A1,A2,B1,B2,C=L[0]
            
            global results
            results = [['',''],['A1 - Assymetry 1',str(A1)], ['A2 - Assymetry 2',str(A2)],['B1 - Borders 1',str(B1)], ['B2 - Borders 2',str(B2)], ['C - Colors' , str(C)]] 
            listDiag=extraction_classification.diagnostic(L)
            global lis
            lis=[]
            for x in listDiag:
                if x:
                    lis.append('Malin')
                else:
                    lis.append('Bénin')
            for i in range(len(results)):
                listBox.insert("", "end", values=(results[i][0], results[i][1]))  #on insère les caractéristiques extraites dans le tableau
                
            Label(scores, text="Linear Discriminant Analysis", font=("Helvetica 16 bold"), justify=LEFT).grid(row=1, columnspan=1,sticky=W)  #on donne les résultats de classification selon différentes méthodes
            Label(scores, text=lis[0], font=("Helvetica 16"), justify=LEFT).grid(row=1, column=1,sticky=W)
            Label(scores, text="Quadratic Discriminant Analysis", font=("Helvetica 16 bold"), justify=LEFT).grid(row=2, columnspan=1,sticky=W)
            Label(scores, text=lis[1], font=("Helvetica 16"), justify=LEFT).grid(row=2, column=1,sticky=W)
            Label(scores, text="Naive Bayes", font=("Helvetica 16 bold"),justify=LEFT).grid(row=3, columnspan=1,sticky=W)
            Label(scores, text=lis[2], font=("Helvetica 16"), justify=LEFT).grid(row=3, column=1,sticky=W)
            #Label(scores, text="K-NearestNeighbours", font=("Helvetica 16 bold"), justify=LEFT).grid(row=4, columnspan=1,sticky=W)
            #Label(scores, text="Malin", font=("Helvetica 16"), justify=LEFT).grid(row=4, column=1,sticky=W)
            Label(scores, text="Logistic Regression", font=("Helvetica 16 bold"), justify=LEFT).grid(row=4, columnspan=1,sticky=W)
            Label(scores, text=lis[3], font=("Helvetica 16"), justify=LEFT).grid(row=4, column=1,sticky=W)

            scores.mainloop()
            scores.destroy()

          
        except NameError:
            im_exists = 'im' in locals() or 'im' in globals()
            im_seg_exists = 'im_seg' in locals() or 'im_seg' in globals()
            if not im_exists:  #ce sont des messages de précaution si aucune image n'a encore été ouverte et qu'on appuie sur le bouton de segmentation ou si on veut extraire les caracs et que la segmentation n'a pas encore été faite
                showerror(title="Erreur", message="Vous n'avez pas ouvert d'image.", default=OK, icon=ERROR)
            elif not im_seg_exists:
                showerror(title="Erreur", message="Vous n'avez pas encore effectué la segmentation.", default=OK, icon=ERROR)

    #la fonction qui s'execute lorsqu'on clique sur le bouton 'Confirmer' de la fenêtre de sauvegarde            
    def save_quit_infos(self):
        try:
            a=direct #si le chemin de sauvegarde n'est pas défini, on a un NameError
            self.last_name=self.e1.get() #on récupère ce qui a été écris dans les champs de texte
            self.first_name=self.e2.get()
            self.age=self.e3.get()
            self.date=self.e4.get()
            self.quit()
        except NameError:
            showerror(title="Erreur", message="Vous n'avez pas défini de lieu de sauvegarde", default=OK, icon=ERROR)
            self.infos.update() #permet de mettre la fenêtre de sauvegarde au premier plan
            self.infos.deiconify()
            
    #la fonction qui s'execute lorsqu'on clique sur le bouton 'Sauvegarder sous' de la fenêtre de sauvegarde            
    def save_as(self):
        global direct
        direct=askdirectory() #on récupère le chemin choisi
        self.infos.textvar.set(str(direct)) #on affiche ce chemin dans le label correspondant
        self.infos.update()   #On remet la fenêtre de sauvegarde au premier plan
        self.infos.deiconify()
        
    #la fonction qui s'execute lorsqu'on clique sur le bouton 'Sauvegarder' de la fenêtre principale              
    def sauvegarder(self):
        try:
            m=im #si une des étapes n'a pas encore été faite, un NameError apparaîtra.
            n=im_seg
            l=results
            self.infos=Tk() #on crée la fenêtre de sauvegarde des informations du patient
            self.infos.title('Informations du patient')
            Label(self.infos, text="Nom", font='Helvetica 18', anchor='w').grid(row=0, sticky=W)
            Label(self.infos, text="Prénom", font='Helvetica 18', anchor='w').grid(row=1, sticky=W)
            Label(self.infos, text="Âge", font='Helvetica 18', anchor='w').grid(row=2, sticky=W)
            Label(self.infos, text="Date", font='Helvetica 18', anchor='w').grid(row=3, sticky=W)
            Label(self.infos, text="    ").grid(row=0,column=1)
            Label(self.infos, text="    ").grid(row=1,column=1)
            Label(self.infos, text="    ").grid(row=2,column=1)
            Label(self.infos, text="    ").grid(row=3,column=1)

            self.e1 = Entry(self.infos, font='Helvetica 18', width=20) #on crée les champs d'entrée de chaque information (Nom, Prénom, Age, ..)
            self.e2 = Entry(self.infos, font='Helvetica 18', width=20)
            self.e3 = Entry(self.infos, font='Helvetica 18', width=20)
            self.e4 = Entry(self.infos, font='Helvetica 18', width=20)
            self.e1.grid(row=0, column=2)
            self.e2.grid(row=1, column=2)
            self.e3.grid(row=2, column=2)
            self.e4.grid(row=3, column=2)
            self.bouton_confirmer=Button(self.infos, text='Confirmer', font='Helvetica 12', command=self.save_quit_infos) #on crée les boutons sauvegarder_sous et confirmer de la fenêtre de sauvegarde
            self.bouton_sauvegarder_sous=Button(self.infos, text='Sauvegarder sous', font='Helvetica 12',command=self.save_as)
            self.bouton_confirmer.grid(row=5, column=0, columnspan=1)
            self.bouton_sauvegarder_sous.grid(row=4, column=0, columnspan=1)
            self.infos.textvar=StringVar(master=self.infos)
            self.savelabel=Label(self.infos,textvariable=self.infos.textvar, width=30, font='Helvetica 12').grid(row=4,column=1, columnspan=2)
            self.infos.mainloop()
            self.infos.destroy()
            
            os.mkdir(direct+'/'+self.last_name)     #On crée un dossier dans le lieu de sauvegarde, au nom du nom de famille
            copyfile(filepath,direct+'/'+self.last_name+'/'+self.last_name+'.png') #on y met l'image initiale en la copiant simplement
            os.chdir(direct+'/'+self.last_name)
            cv2.imwrite(self.last_name+'_seg.png', cv2.cvtColor(im, cv2.COLOR_RGB2BGR)) #on met aussi l'image segmentée, en faisant attention au fait que imwrite nécessite une image en BGR
           
            cv2.imwrite(self.last_name+'_mask.png', cv2.cvtColor(255*im_seg, cv2.COLOR_GRAY2BGR)) #on met le masque
            f=open(self.last_name+'.txt','w+') #on met un fichier texte contant les données du patient et les résultats
            f.write('********** Résultats **********\n\n')
            f.write('Nom : '+self.last_name+'\nPrénom : '+self.first_name+'\nÂge : '+self.age+' ans\nDate : '+self.date+'\n\n')
            for r in results:
                f.write(r[0]+' : '+r[1]+'\n')
            f.write('\nLinear Discriminant Analysis : '+lis[0]+'\nQuadratic Discriminant Analysis : '+lis[1]+'\nNaive Bayes: '+lis[2]+'\nLogistic Regression : '+lis[3]+'\n')
            f.close()
        except NameError:
            im_exists = 'im' in locals() or 'im' in globals()
            im_seg_exists = 'im_seg' in locals() or 'im_seg' in globals()
            im_extr_exists = 'results' in locals() or 'results' in globals()
            if not im_exists:
                showerror(title="Erreur", message="Vous n'avez pas ouvert d'image.", default=OK, icon=ERROR)
            elif not im_seg_exists:
                showerror(title="Erreur", message="Vous n'avez pas encore effectué la segmentation.", default=OK, icon=ERROR)
            elif not im_extr_exists:
                showerror(title="Erreur", message="Vous n'avez pas encore extrait les caractéristiques de l'image.", default=OK, icon=ERROR)

       

fenetre = Tk()
fenetre.title("Détecteur de mélanomes")
w, h = 0.75*fenetre.winfo_screenwidth(), fenetre.winfo_screenheight()
fenetre.geometry("%dx%d+0+0" % (w, h))

interface = Interface(fenetre)
interface.mainloop()
interface.destroy()

from tkinter import *
from tkinter import messagebox, font
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # feature scaling
from sklearn.metrics import accuracy_score  # test the model with the same training data and also check its accuracy
from sklearn.metrics import classification_report
from joblib import dump
from PIL import ImageTk


# myFont = font.Font(family='Helvetica', size=20, weight="bold")

def about():
    # root3 = Tk()
    root3 = Toplevel(root)
    root3.geometry('600x500')
    root3.title("About-Us")
    root3.resizable(False, False)
    root3.iconbitmap("about-us.ico")
    bgm_img = PhotoImage(file="pic1.png")
    img31 = Label(root3, image=bgm_img)
    img31.pack()
    valid_var = "WE ARE THE STUDENTS OF MEERUT INSTITUTE OF TECHNOLOGY, MEERUT."
    valid_var1 = "We are developing this project to help persons to check \nwhether they have BREAST CANCER or not.\nUser provide some values to the model and \nmodel predict the diagnosis.\nThis model have 95% + accuracy."
    valid_var2 = " Prachi Sharma "
    valid_var3 = " Prateek Goel "
    valid_var4 = " Prerna Chaudhary "
    valid_var5 = " Shubhangi Bhardwaj "
    img123 = PhotoImage(file='bdia.png')
    panel1 = Label(root3, image=img123)
    panel1.place(x=10, y=4)
    myFont1 = font.Font(family='Helvetica', size=20, weight="bold")
    myFont2 = font.Font(family='Helvetica', size=11, weight="bold")
    myFont3 = font.Font(family='Helvetica', size=13, weight="bold")

    Label(root3, text="ABOUT - US", bg='white', fg='red', font=myFont1).place(x=210, y=115)
    Label(root3, text=valid_var, bg='white', font=myFont2).place(x=11, y=165)
    Label(root3, text=valid_var1.upper(), bg='white', font=myFont3).place(x=10, y=215)
    Label(root3, text=valid_var2.upper(), bg='white', font=myFont3).place(x=370, y=330)
    Label(root3, text=valid_var3.upper(), bg='white', font=myFont3).place(x=370, y=370)
    Label(root3, text=valid_var4.upper(), bg='white', font=myFont3).place(x=370, y=410)
    Label(root3, text=valid_var5.upper(), bg='white', font=myFont3).place(x=370, y=450)
    # lb.place(x=250, y=250)
    # btn35 = Button(root2, text="About Us.", font=15, width=10, height=2, command="")
    # btn35.place(x=200, y=20)
    # root3.mainloop()
    root3.mainloop()


def symptoms():
    # 4
    root4 = Toplevel(root)
    root4.geometry('600x550')
    root4.title("Symptoms")
    root4.resizable(False, False)
    root4.iconbitmap("helpline-1.ico")
    bgm_img = PhotoImage(file="pic1.png")
    img31 = Label(root4, image=bgm_img)
    img31.pack()
    valid_var = "  Signs and symptoms of breast cancer may include:"
    valid_var1 = "A breast lump or thickening that feels different from the surrounding tissue"
    valid_var2 = "Change in the size, shape or appearance of a breast"
    valid_var3 = "Changes to the skin over the breast, such as dimpling"
    valid_var4 = "A newly inverted nipple"
    valid_var5 = "Peeling, scaling, crusting or flaking of the pigmented area of skin \nsurrounding the nipple (areola) or breast skin"
    img123 = PhotoImage(file='bdia.png')
    panel1 = Label(root4, image=img123)
    panel1.place(x=10, y=4)
    myFont1 = font.Font(family='Helvetica', size=24, weight="bold")
    myFont2 = font.Font(family='Helvetica', size=15, weight="bold")
    myFont3 = font.Font(family='Helvetica', size=13)

    Label(root4, text="SYMPTOMS", fg='red', font=myFont1).place(x=210, y=120)
    Label(root4, text=valid_var, bg='white', font=myFont2).place(x=25, y=200)
    Label(root4, text=valid_var1, bg='white', font=myFont3).place(x=8, y=260)
    Label(root4, text=valid_var2, bg='white', font=myFont3).place(x=8, y=315)
    Label(root4, text=valid_var3, bg='white', font=myFont3).place(x=8, y=370)
    Label(root4, text=valid_var4, bg='white', font=myFont3).place(x=8, y=425)

    Label(root4, text=valid_var5, bg='white', font=myFont3).place(x=8, y=480)

    root4.mainloop()


def cause_and_risk():
    # 5
    # root2 = Tk()

    # root5 = Tk()
    root5 = Toplevel(root)
    root5.geometry('600x550')
    root5.title("Cause & Risk Factor")
    root5.resizable(False, False)
    root5.iconbitmap("helpline-1.ico")
    bgm_img = PhotoImage(file="pic1.png")
    img31 = Label(root5, image=bgm_img)
    img31.pack()
    valid_var = "Factors that are associated with an increased risk of \nbreast cancer include:"
    valid_var1 = "Being female. Women are much more likely than men are to \ndevelop breast cancer."
    valid_var2 = "A personal history of breast conditions. If you've had a \nbreast biopsy that found lobular carcinoma in situ (LCIS) or a typical \nhyperplasia of the breast, you have an increased risk of breast cancer."
    valid_var3 = "Changes to the skin over the breast, such as dimpling"
    valid_var4 = "A newly inverted nipple"
    valid_var5 = "Peeling, scaling, crusting or flaking of the pigmented area \nof skin surrounding the nipple (areola) or breast skin"
    img123 = PhotoImage(file='bdia.png')
    panel1 = Label(root5, image=img123)
    panel1.place(x=10, y=4)
    myFont1 = font.Font(family='Helvetica', size=24, weight="bold")
    myFont2 = font.Font(family='Helvetica', size=15, weight="bold")
    myFont3 = font.Font(family='Helvetica', size=13)

    Label(root5, text="CAUSE & RISK FACTOR", fg='red', font=myFont1).place(x=120, y=120)
    Label(root5, text=valid_var, bg='white', font=myFont2).place(x=2, y=195)
    Label(root5, text=valid_var1, bg='white', font=myFont3).place(x=8, y=260)
    Label(root5, text=valid_var2, bg='white', font=myFont3).place(x=8, y=315)
    Label(root5, text=valid_var3, bg='white', font=myFont3).place(x=8, y=390)
    Label(root5, text=valid_var4, bg='white', font=myFont3).place(x=8, y=430)

    Label(root5, text=valid_var5, bg='white', font=myFont3).place(x=8, y=470)

    root5.mainloop()


def treatment():
    # 7
    # from tkinter import *
    # from tkinter import messagebox, font
    # root7 = Tk()
    root7 = Toplevel(root)
    root7.geometry('600x550')
    root7.title("Treatment")
    root7.resizable(False, False)
    root7.iconbitmap("helpline-1.ico")
    bgm_img = PhotoImage(file="pic1.png")
    img31 = Label(root7, image=bgm_img)
    img31.pack()
    valid_var = "Operations used to treat breast cancer include:"
    valid_var1 = "Removing the breast cancer (lumpectomy)."
    valid_var2 = "Removing the entire breast (mastectomy). "
    valid_var3 = "Removing a limited number of lymph nodes (sentinel node biopsy)."
    valid_var4 = "Removing several lymph nodes (axillary lymph node dissection)"
    valid_var5 = "Removing both breasts. Removing both breasts. "
    img123 = PhotoImage(file='bdia.png')
    panel1 = Label(root7, image=img123)
    panel1.place(x=10, y=4)
    myFont1 = font.Font(family='Helvetica', size=24, weight="bold")
    myFont2 = font.Font(family='Helvetica', size=15, weight="bold")
    myFont3 = font.Font(family='Helvetica', size=13)

    Label(root7, text="TREATMENT", fg='red', font=myFont1).place(x=200, y=120)
    Label(root7, text=valid_var, bg='white', font=myFont2).place(x=25, y=200)
    Label(root7, text=valid_var1, bg='white', font=myFont3).place(x=8, y=260)
    Label(root7, text=valid_var2, bg='white', font=myFont3).place(x=8, y=315)
    Label(root7, text=valid_var3, bg='white', font=myFont3).place(x=8, y=370)
    Label(root7, text=valid_var4, bg='white', font=myFont3).place(x=8, y=425)

    Label(root7, text=valid_var5, bg='white', font=myFont3).place(x=8, y=480)

    root7.mainloop()


def preventions():
    # 6
    # from tkinter import *
    # from tkinter import messagebox, font
    # root6 = Tk()
    root6 = Toplevel(root)
    root6.geometry('600x550')
    root6.title("Preventions")
    root6.resizable(False, False)
    root6.iconbitmap("helpline-1.ico")
    bgm_img = PhotoImage(file="pic1.png")
    img31 = Label(root6, image=bgm_img)
    img31.pack()
    valid_var = "Breast cancer risk reduction for women with an average risk"
    valid_var1 = "Become familiar with your breasts through breast self-exam for breast awareness."
    valid_var2 = "Drink alcohol in moderation, if at all. Limit the amount of alcohol \nyou drink to no more than one drink a day, if you choose to drink."
    valid_var3 = "Exercise most days of the week. Aim for at least 30 minutes of exercise \non most days of the week. "
    valid_var4 = "Limit postmenopausal hormone therapy. Combination hormone therapy may \nincrease the risk of breast cancer."
    valid_var5 = "Maintain a healthy weight. If your weight is healthy, work to maintain that weight."
    img123 = PhotoImage(file='bdia.png')
    panel1 = Label(root6, image=img123)
    panel1.place(x=10, y=4)
    myFont1 = font.Font(family='Helvetica', size=24, weight="bold")
    myFont2 = font.Font(family='Helvetica', size=15, weight="bold")
    myFont3 = font.Font(family='Helvetica', size=13)

    Label(root6, text="PREVENTIONS", fg='red', font=myFont1).place(x=190, y=120)
    Label(root6, text=valid_var, bg='white', font=myFont2).place(x=10, y=200)
    Label(root6, text=valid_var1, bg='white', font=myFont3).place(x=8, y=260)
    Label(root6, text=valid_var2, bg='white', font=myFont3).place(x=8, y=315)
    Label(root6, text=valid_var3, bg='white', font=myFont3).place(x=8, y=370)
    Label(root6, text=valid_var4, bg='white', font=myFont3).place(x=8, y=425)

    Label(root6, text=valid_var5, bg='white', font=myFont3).place(x=8, y=480)

    root6.mainloop()


def helpline():
    # root2 = Tk()
    root2 = Toplevel(root)
    root2.geometry('600x550')
    root2.title("Helpline")
    root2.resizable(False, False)
    root2.iconbitmap("helpline-1.ico")
    bgm_img = PhotoImage(file="pic1.png")
    img31 = Label(root2, image=bgm_img)
    img31.pack()
    valid_var = "  What can You do to reduce your risk of breast cancer?"
    valid_var1 = "Research shows that lifestyle changes can decrease the risk of breast cancer,\n even in women at high risk.To lower your risk follow below points : "
    valid_var2 = "Limit alcohol : The more alcohol you drink, the greater your risk of developing \n breast cancer."
    valid_var3 = "Maintain a healthy weight : If your weight is healthy, work to maintain that weight.\nIf you need to lose weight, ask your doctor about that."
    valid_var4 = "Be physically active : Physical activity can help you maintain a healthy weight,\n which helps prevent breast cancer."
    valid_var5 = "Breast-feed : Breast-feeding might play a role in breast cancer prevention. \nThe longer you breast-feed, the greater the protective effect."
    img123 = PhotoImage(file='bdia.png')
    panel1 = Label(root2, image=img123)
    panel1.place(x=10, y=4)
    myFont1 = font.Font(family='Helvetica', size=24, weight="bold")
    myFont2 = font.Font(family='Helvetica', size=15, weight="bold")
    myFont3 = font.Font(family='Helvetica', size=13)

    Label(root2, text="Helpline Number : 1800-2700-703", fg='red', font=myFont1).place(x=50, y=120)
    Label(root2, text=valid_var, bg='white', font=myFont2).place(x=25, y=200)
    Label(root2, text=valid_var1, bg='white', font=myFont3).place(x=8, y=260)
    Label(root2, text=valid_var2, bg='white', font=myFont3).place(x=8, y=315)
    Label(root2, text=valid_var3, bg='white', font=myFont3).place(x=8, y=370)
    Label(root2, text=valid_var4, bg='white', font=myFont3).place(x=8, y=425)
    Label(root2, text=valid_var5, bg='white', font=myFont3).place(x=8, y=480)

    root2.mainloop()


def get_started():
    def close():
        root1.destroy()

    root1 = Toplevel(root)

    # panel.grid(column=0, row=1)
    # panel.place(x=8100, y=250)
    # panel.pack(side="top", fill='both', expand="yes")
    bg_img = PhotoImage(file="pic1.png")
    img3 = Label(root1, image=bg_img)
    img3.place(x=0, y=0)

    img = PhotoImage(file='bdia.png')
    panel = Label(root1, image=img)
    panel.place(x=250, y=4)
    root1.iconbitmap("form.ico")

    # # Create Canvas
    # canvas1 = Canvas(root1, width=600,
    #                  height=700)
    #
    # canvas1.pack(fill="both", expand=True)
    #
    # # Display image
    # canvas1.create_image(0, 0, image=bg,
    #                      anchor="nw")

    def code_ibm(name1_info, name2_info, name3_info, name4_info, name5_info, name6_info, name7_info, name8_info,
                 name9_info, name10_info, name11_info, name12_info, name13_info, name14_info, name15_info, name16_info,
                 name17_info, name18_info, name19_info, name20_info, name21_info, name22_info, name23_info, name24_info,
                 name25_info, name26_info, name27_info, name28_info, name29_info):
        # using panda module read csv file
        dataframe = pd.read_csv("data.csv")

        # In[3]:

        # print(dataframe.head())  # getting only 5 values of csv file

        # In[4]:

        # print(
        #     dataframe.info())  # checking which attribute have non-null values so that we can remove those columns which are having a lot of null value

        # In[5]:

        # print(dataframe.isna().sum())  # show count of null values in attributes

        # In[6]:

        # print(dataframe.shape)  # show size of dataset

        # In[7]:

        dataframe = dataframe.dropna(axis=1)  # remove column of null value

        # In[8]:

        # print(dataframe.shape)

        # In[9]:

        # print(
        #     dataframe.describe())  # show the count ,mean ,standard deviation ,minimum value and maximum values of the data

        # In[10]:

        # print(dataframe['diagnosis'].value_counts())  # Get the count of malignant<M> and Benign<B> cells

        # In[11]:

        sns.countplot(dataframe['diagnosis'],
                      label="count")  # seaborn library will graphically show malignant and begnin ratio

        # In[12]:

        # label encoding(convert the value of M and B into 1 and 0)

        labelencoder_Y = LabelEncoder()
        dataframe.iloc[:, 1] = labelencoder_Y.fit_transform(dataframe.iloc[:, 1].values)

        # In[13]:

        # print(dataframe.head())  # value of diagnosis column changed from M and B to 1 and 0

        # In[14]:

        sns.pairplot(dataframe.iloc[:, 1:5], hue="diagnosis")

        # In[15]:

        dataframe.iloc[:, 1:32].corr()
        # give the correlation means how much the value any attribute is affecting the value of another attribute

        # In[16]:

        # plt.figure(figsize=(10, 10))
        # sns.heatmap(dataframe.iloc[:, 1:10].corr(), annot=True, fmt=".0%")  # visualize the correlation

        # In[17]:

        X = dataframe.iloc[:, 2:31].values  # split the dataset into dependent(X) and Independent(Y) datasets
        Y = dataframe.iloc[:, 1].values

        # In[18]:

        # splitting the data into training and test dateset

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

        # In[19]:

        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit_transform(X_test)

        # In[20]:

        # models/ Algorithms
        # among these three model we can select anyone and obsviously we will select with higher accuracy

        def models(X_train, Y_train):
            # logistic regression
            # from sklearn.linear_model import LogisticRegression
            # log = LogisticRegression(random_state=0)
            # log.fit(X_train, Y_train)
            #
            # # Decision Tree
            # from sklearn.tree import DecisionTreeClassifier
            # tree = DecisionTreeClassifier(random_state=0, criterion="entropy")
            # tree.fit(X_train, Y_train)
            tree = 5
            log = 5

            # Random Forest
            from sklearn.ensemble import RandomForestClassifier
            forest = RandomForestClassifier(random_state=0, criterion="entropy", n_estimators=10)
            forest.fit(X_train, Y_train)

            # print('[0]logistic regression accuracy:',
            #       log.score(X_train, Y_train))  # will show the accuracy of this algorithm over this data
            # print('[1]Decision tree accuracy:', tree.score(X_train, Y_train))
            # print('[2]Random forest accuracy:', forest.score(X_train, Y_train))

            return log, tree, forest

        # In[21]:

        model = models(X_train, Y_train)  # decision tree is overfitted

        # In[22]:

        # for i in range(len(model)):
        #     print("Model", i)
        #     print(classification_report(Y_test, model[i].predict(X_test)))
        #     print('Accuracy : ', accuracy_score(Y_test, model[i].predict(X_test)))

        # # Testing over the extracted data from the already known dataset

        # In[23]:

        # prediction with random-forest model over the testing data
        # pred = model[2].predict(X_test)
        # print('Predicted values:')
        # print(pred)
        # print('Actual values:')
        # print(Y_test)

        # In[40]:

        radius_mean = float(name1_info)
        texture_mean = float(name2_info)
        perimeter_mean = float(name3_info)
        area_mean = float(name4_info)
        smoothness_mean = float(name5_info)
        compactness_mean = float(name6_info)
        concavity_mean = float(name7_info)
        concave_points_mean = float(name8_info)
        symmetry_mean = float(name9_info)
        fractal_diamension_mean = float(name10_info)
        Radius_se = float(name11_info)
        Texture_se = float(name12_info)
        perimeter_se = float(name13_info)
        area_se = float(name14_info)
        smoothness_se = float(name15_info)
        compactness_se = float(name16_info)
        concavity_se = float(name17_info)
        concave_points_se = float(name18_info)
        symmetry_se = float(name19_info)
        Fractal_dimension_se = float(name20_info)
        texture_worst = float(name21_info)
        perimeter_worst = float(name22_info)
        area_worst = float(name23_info)
        smoothness_worst = float(name24_info)
        compactness_worst = float(name25_info)
        concavity_worst = float(name26_info)
        concave_points_worst = float(name27_info)
        symmetry_worst = float(name28_info)
        fractal_dimension_worst = float(name29_info)

        # # Model is ready to predict over input data

        # In[47]:

        testing = [
            [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean,
             concave_points_mean, symmetry_mean, fractal_diamension_mean, Radius_se, Texture_se, perimeter_se,
             area_se,
             smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, Fractal_dimension_se,
             texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst,
             concave_points_worst, symmetry_worst, fractal_dimension_worst]]
        print(testing)

        # In[45]:

        predicted = model[2].predict(testing)
        if predicted == [1]:
            print(
                "We are sorry!\nBut Unfortunately you have Breast Cancer.\nCheck Helpline Number and Take Precaution\n")
        else:
            print("Fortunately..According to all the value you Do not have Breast Cancer\n VOLLAA!!!\n")

        return predicted

    def isChecked():
        name1_info = name1Value.get()
        name2_info = name2Value.get()
        name3_info = name3Value.get()
        name4_info = name4Value.get()
        name5_info = name5Value.get()
        name6_info = name6Value.get()
        name7_info = name7Value.get()
        name8_info = name8Value.get()
        name9_info = name9Value.get()
        name10_info = name10Value.get()
        name11_info = name11Value.get()
        name12_info = name12Value.get()
        name13_info = name13Value.get()
        name14_info = name14Value.get()
        name15_info = name15Value.get()
        name16_info = name16Value.get()
        name17_info = name17Value.get()
        name18_info = name18Value.get()
        name19_info = name19Value.get()
        name20_info = name20Value.get()
        name21_info = name21Value.get()
        name22_info = name22Value.get()
        name23_info = name23Value.get()
        name24_info = name24Value.get()
        name25_info = name25Value.get()
        name26_info = name26Value.get()
        name27_info = name27Value.get()
        name28_info = name28Value.get()
        name29_info = name29Value.get()
        # print(name1_info)
        if checkValue.get():
            if ((len(name1_info) == 0) or (len(name2_info) == 0) or (len(name3_info) == 0) or (
                    len(name4_info) == 0) or (
                    len(name5_info) == 0) or (len(name6_info) == 0) or (len(name7_info) == 0) or (
                    len(name8_info) == 0) or (len(name9_info) == 0) or (len(name10_info) == 0) or (
                    len(name11_info) == 0) or (len(name12_info) == 0) or (len(name13_info) == 0) or (
                    len(name14_info) == 0) or (
                    len(name15_info) == 0) or (len(name16_info) == 0) or (len(name17_info) == 0) or (
                    len(name18_info) == 0) or (len(name19_info) == 0) or (len(name20_info) == 0) or (
                    len(name21_info) == 0) or (len(name22_info) == 0) or (len(name23_info) == 0) or (
                    len(name24_info) == 0) or (
                    len(name25_info) == 0) or (len(name26_info) == 0) or (len(name27_info) == 0) or (
                    len(name28_info) == 0) or (len(name29_info) == 0)):

                messagebox.showerror("Warning", "The Entry Filled Can not be Empty")
                check.toggle()
                # break
                # checkValue.set(1)

            else:
                btn['state'] = NORMAL
                btn.configure(text='Submit Details')
        elif not checkValue.get():
            # if
            btn['state'] = DISABLED
            btn.configure(text='Submit Details')
        else:
            messagebox.showerror('PythonGuides', 'Something went wrong!')

    def submit():
        # print("Details Submitted")
        name1_info = name1Value.get()
        name2_info = name2Value.get()
        name3_info = name3Value.get()
        name4_info = name4Value.get()
        name5_info = name5Value.get()
        name6_info = name6Value.get()
        name7_info = name7Value.get()
        name8_info = name8Value.get()
        name9_info = name9Value.get()
        name10_info = name10Value.get()
        name11_info = name11Value.get()
        name12_info = name12Value.get()
        name13_info = name13Value.get()
        name14_info = name14Value.get()
        name15_info = name15Value.get()
        name16_info = name16Value.get()
        name17_info = name17Value.get()
        name18_info = name18Value.get()
        name19_info = name19Value.get()
        name20_info = name20Value.get()
        name21_info = name21Value.get()
        name22_info = name22Value.get()
        name23_info = name23Value.get()
        name24_info = name24Value.get()
        name25_info = name25Value.get()
        name26_info = name26Value.get()
        name27_info = name27Value.get()
        name28_info = name28Value.get()
        name29_info = name29Value.get()
        # print(name1_info)
        Label(root1, text="Details Submitted", fg="green", font=11).place(x=700, y=640)
        value = code_ibm(name1_info, name2_info, name3_info, name4_info, name5_info, name6_info, name7_info, name8_info,
                         name9_info, name10_info, name11_info, name12_info, name13_info, name14_info, name15_info,
                         name16_info,
                         name17_info, name18_info, name19_info, name20_info, name21_info, name22_info, name23_info,
                         name24_info,
                         name25_info, name26_info, name27_info, name28_info, name29_info)
        # root1.option_add('*Dialog.msg.font', 'Helvetica 12')
        if value == 1:
            messagebox.showinfo(title='Result',
                                message="\tWe are sorry!\t\nBut Unfortunately you have Breast Cancer.\nCheck Helpline Number and Take Precaution\n")
        else:
            messagebox.showinfo(title='Result', message="Congrats,you don't have Breast Cancer!!")
        # root1.option_clear()

    root1.title("Breast Cancer Detection")
    # photo = PhotoImage(file="imgg.ico")
    # root1.iconbitmap("bc.ico")

    root1.geometry('1100x700')

    root1.resizable(False, False)

    # Label(text="Breast Cancer Detection", font='arial 30').place(x=80, y=25)

    Label(root1, text='Radius Mean', font=15).place(x=30, y=100)
    Label(root1, text='Texture Mean', font=15).place(x=30, y=150)
    Label(root1, text='Perimeter Mean', font=15).place(x=30, y=200)
    Label(root1, text='Area Mean', font=15).place(x=30, y=250)
    Label(root1, text='Smoothness Mean', font=15).place(x=30, y=300)
    Label(root1, text='Compactness Mean', font=15).place(x=30, y=350)
    Label(root1, text='Concavity Mean', font=15).place(x=30, y=400)
    Label(root1, text='Concave Points Mean', font=15).place(x=30, y=450)
    Label(root1, text='Symmetry Mean', font=15).place(x=30, y=500)
    Label(root1, text='Fractal Dimension Mean', font=10).place(x=100, y=550)
    # root1,
    Label(root1, text='Radius Size', font=15).place(x=370, y=100)
    Label(root1, text='Texture SE', font=25).place(x=370, y=150)
    Label(root1, text='Perimeter SE', font=25).place(x=370, y=200)
    Label(root1, text='Area SE', font=25).place(x=370, y=250)
    Label(root1, text='Smoothness SE', font=25).place(x=370, y=300)
    Label(root1, text='Compactness SE', font=25).place(x=370, y=350)
    Label(root1, text='Concavity SE', font=25).place(x=370, y=400)
    Label(root1, text='Concave Point SE', font=25).place(x=370, y=450)
    Label(root1, text='Symmetry SE', font=25).place(x=370, y=500)
    Label(root1, text='Fractal Dimension SE', font=25).place(x=500, y=550)
    # root1,
    Label(root1, text='Texture Worst', font=25).place(x=710, y=100)
    Label(root1, text='Perimeter Worst', font=25).place(x=710, y=150)
    Label(root1, text='Area Worst', font=25).place(x=710, y=200)
    Label(root1, text='Smoothness Worst', font=25).place(x=710, y=250)
    Label(root1, text='Compactness Worst', font=25).place(x=710, y=300)
    Label(root1, text='Concavity Worst', font=25).place(x=710, y=350)
    Label(root1, text='Concave Point Worst', font=25).place(x=710, y=400)
    Label(root1, text='Symmetry Worst', font=25).place(x=710, y=450)
    Label(root1, text='Fractal Dimension Worst', font=25).place(x=710, y=500)

    # Label(text='Class', font=25).place(x=50, y=550)
    # Label(text='Name', font=25).place(x=100, y=500)
    # Label(text='Name', font=25).place(x=100, y=540)
    # Label(text='Name', font=25).place(x=100, y=580)

    # input

    name1Value = StringVar()
    name2Value = StringVar()
    name3Value = StringVar()
    name4Value = StringVar()
    name5Value = StringVar()
    name6Value = StringVar()
    name7Value = StringVar()
    name8Value = StringVar()
    name9Value = StringVar()
    name10Value = StringVar()
    name11Value = StringVar()
    name12Value = StringVar()
    name13Value = StringVar()
    name14Value = StringVar()
    name15Value = StringVar()
    name16Value = StringVar()
    name17Value = StringVar()
    name18Value = StringVar()
    name19Value = StringVar()
    name20Value = StringVar()
    name21Value = StringVar()
    name22Value = StringVar()
    name23Value = StringVar()
    name24Value = StringVar()
    name25Value = StringVar()
    name26Value = StringVar()
    name27Value = StringVar()
    name28Value = StringVar()
    name29Value = StringVar()

    nameEntry = Entry(root1, textvariable=name1Value, width=15, bd=2, font=20).place(x=205, y=100)
    nameEntry = Entry(root1, textvariable=name2Value, width=15, bd=2, font=20).place(x=205, y=150)
    nameEntry = Entry(root1, textvariable=name3Value, width=15, bd=2, font=20).place(x=205, y=200)
    nameEntry = Entry(root1, textvariable=name4Value, width=15, bd=2, font=20).place(x=205, y=250)
    nameEntry = Entry(root1, textvariable=name5Value, width=15, bd=2, font=20).place(x=205, y=300)
    nameEntry = Entry(root1, textvariable=name6Value, width=15, bd=2, font=20).place(x=205, y=350)
    nameEntry = Entry(root1, textvariable=name7Value, width=15, bd=2, font=20).place(x=205, y=400)
    nameEntry = Entry(root1, textvariable=name8Value, width=15, bd=2, font=20).place(x=205, y=450)
    nameEntry = Entry(root1, textvariable=name9Value, width=15, bd=2, font=20).place(x=205, y=500)
    nameEntry = Entry(root1, textvariable=name10Value, width=15, bd=2, font=20).place(x=300, y=550)

    nameEntry = Entry(root1, textvariable=name11Value, width=15, bd=2, font=20).place(x=545, y=100)
    nameEntry = Entry(root1, textvariable=name12Value, width=15, bd=2, font=20).place(x=545, y=150)
    nameEntry = Entry(root1, textvariable=name13Value, width=15, bd=2, font=20).place(x=545, y=200)
    nameEntry = Entry(root1, textvariable=name14Value, width=15, bd=2, font=20).place(x=545, y=250)
    nameEntry = Entry(root1, textvariable=name15Value, width=15, bd=2, font=20).place(x=545, y=300)
    nameEntry = Entry(root1, textvariable=name16Value, width=15, bd=2, font=20).place(x=545, y=350)
    nameEntry = Entry(root1, textvariable=name17Value, width=15, bd=2, font=20).place(x=545, y=400)
    nameEntry = Entry(root1, textvariable=name18Value, width=15, bd=2, font=20).place(x=545, y=450)
    nameEntry = Entry(root1, textvariable=name19Value, width=15, bd=2, font=20).place(x=545, y=500)
    nameEntry = Entry(root1, textvariable=name20Value, width=15, bd=2, font=20).place(x=700, y=550)
    #
    nameEntry = Entry(root1, textvariable=name21Value, width=15, bd=2, font=20).place(x=900, y=100)
    nameEntry = Entry(root1, textvariable=name22Value, width=15, bd=2, font=20).place(x=900, y=150)
    nameEntry = Entry(root1, textvariable=name23Value, width=15, bd=2, font=20).place(x=900, y=200)
    nameEntry = Entry(root1, textvariable=name24Value, width=15, bd=2, font=20).place(x=900, y=250)
    nameEntry = Entry(root1, textvariable=name25Value, width=15, bd=2, font=20).place(x=900, y=300)
    nameEntry = Entry(root1, textvariable=name26Value, width=15, bd=2, font=20).place(x=900, y=350)
    nameEntry = Entry(root1, textvariable=name27Value, width=15, bd=2, font=20).place(x=900, y=400)
    nameEntry = Entry(root1, textvariable=name28Value, width=15, bd=2, font=20).place(x=900, y=450)
    nameEntry = Entry(root1, textvariable=name29Value, width=15, bd=2, font=20).place(x=900, y=500)
    # nameEntry = Entry(root1, textvariable=name10Value, width=15, bd=2, font=20).place(x=900, y=550)

    # nameEntry.place(x=200,y=100)
    # button

    checkValue = BooleanVar()
    checkValue.set(False)
    check = Checkbutton(root1, text="Are you sure?", variable=checkValue, onvalue=True, offvalue=False,
                        command=isChecked)
    check.place(x=500, y=590)

    btn = Button(root1, text="Submit Details", font=20, state=DISABLED, width=13, height=2, command=submit)
    btn.place(x=485, y=630)

    root1.mainloop()


root = Tk()
root.title("Breast Cancer Diagnosis")
# root.geometry('600x500')
root.geometry('1030x700')
root.iconbitmap("breast.ico")
root.resizable(False, False)
bg_img = PhotoImage(file="pic1.png")
img3 = Label(root, image=bg_img)
img3.pack()

img = PhotoImage(file='bdia.png')
panel = Label(root, image=img)
panel.place(x=219, y=8)
myFont = font.Font(family='Helvetica', size=25, weight="bold")
myFont_head = font.Font(family='Arial', size=15, weight="bold")
# 3 diagnosis
btn1 = Button(text="Diagnosis", font=myFont_head, fg='purple', width=20, height=2, command=get_started)
btn1.place(x=370, y=290)

# 6 helpline
btn2 = Button(text="HelpLine No.", font=myFont_head, fg='purple', width=20, height=2, command=helpline)
# btn2.place(x=420, y=350)
btn2.place(x=370, y=530)
# 7 about
btn3 = Button(text="About Us.", font=myFont_head, fg='purple', width=20, height=2, command=about)
btn3.place(x=370, y=610)

# 4 treatment
btn4 = Button(text="Treatment", font=myFont_head, fg='purple', width=20, height=2, command=treatment)
# btn2.place(x=420, y=350)
btn4.place(x=370, y=370)

# 5 prevention

btn5 = Button(text="Preventions", font=myFont_head, fg='purple', width=20, height=2, command=preventions)
btn5.place(x=370, y=450)

# 2 cause and risk factor
btn6 = Button(text="Cause & Risk Factor", font=myFont_head, fg='purple', width=20, height=2, command=cause_and_risk)
# btn2.place(x=420, y=350)
btn6.place(x=370, y=210)

# 1 symptoms
btn7 = Button(text="Symptoms", font=myFont_head, fg='purple', width=20, height=2, command=symptoms)
btn7.place(x=370, y=130)

# images


about_img = PhotoImage(file='about_us6.png')
about_panel = Label(root, image=about_img)
about_panel.place(x=300, y=610)

symptoms_img = PhotoImage(file='symptoms.png')
symptoms_panel = Label(root, image=symptoms_img)
symptoms_panel.place(x=300, y=130)

cause_img = PhotoImage(file='risk_factor.png')
cause_panel = Label(root, image=cause_img)
cause_panel.place(x=300, y=210)

diagnosis_img = PhotoImage(file='diagnosis.png')
diagnosis_panel = Label(root, image=diagnosis_img)
diagnosis_panel.place(x=300, y=290)

treatment_img = PhotoImage(file='treatment.png')
treatment_panel = Label(root, image=treatment_img)
treatment_panel.place(x=300, y=370)

prevention_img = PhotoImage(file='prevention.png')
prevention_panel = Label(root, image=prevention_img)
prevention_panel.place(x=300, y=450)

helpline_img = PhotoImage(file='helpline.png')
helping_panel = Label(root, image=helpline_img)
helping_panel.place(x=300, y=530)

root.mainloop()

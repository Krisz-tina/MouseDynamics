import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtWidgets import QMainWindow, QLabel
from PyQt5.QtWidgets import QGridLayout, QWidget, QDesktopWidget
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (QWidget, QLabel, QComboBox, QApplication, QFormLayout, QVBoxLayout, QStackedLayout, QPushButton, QHBoxLayout, QLineEdit)
from PyQt5 import QtGui
from user_interface import verifying_session
from user_interface import plotting_session


class UserInterface(QWidget):

    def __init__(self):
        super().__init__()
        self.combo_container_layout = QStackedLayout()
        self.combo_container = QWidget()
        self.button_verify = QPushButton('Verify')
        self.label_image = QLabel(self)
        self.label_cross_grey = QLabel(self)
        self.label_tick_grey = QLabel(self)
        self.label_threshold = QLabel(self)
        self.label_score = QLabel(self)
        self.text_threshold = QLineEdit(self)
        self.text_threshold.hide()
        self.text_score = QLineEdit(self)
        self.text_score.hide()
        self.label_cross = QLabel(self)
        self.label_tick = QLabel(self)
        self.initUI()

    def initUI(self):
        self.combo_user = QComboBox(self)
        self.combo_user.addItem('user7')
        self.combo_user.addItem('user9')
        self.combo_user.addItem('user12')
        self.combo_user.addItem('user15')
        self.combo_user.addItem('user16')
        self.combo_user.addItem('user20')
        self.combo_user.addItem('user21')
        self.combo_user.addItem('user23')
        self.combo_user.addItem('user29')
        self.combo_user.addItem('user35')
        self.combo_user.move(50, 50)
        self.combo_user.currentIndexChanged[int].connect(self.indexChangedA)

        self.combo_type = QComboBox(self)
        self.combo_type.addItem('positive')
        self.combo_type.addItem('negative')
        self.combo_type.move(50, 100)
        self.combo_type.currentIndexChanged[int].connect(self.indexChangedB)

        self.combo7pos = QComboBox(self)
        self.combo7pos.addItem('session_5289449664')
        self.combo7pos.addItem('session_5528609206')
        self.combo7pos.currentIndexChanged.connect(self.indexChangedB)

        self.combo7neg = QComboBox(self)
        self.combo7neg.addItem('session_0147719489')
        self.combo7neg.addItem('session_5483480261')
        self.combo7neg.currentIndexChanged.connect(self.indexChangedB)

        self.combo9pos = QComboBox(self)
        self.combo9pos.addItem('session_3926840201')
        self.combo9pos.addItem('session_4243186683')
        self.combo9pos.currentIndexChanged.connect(self.indexChangedB)

        self.combo9neg = QComboBox(self)
        self.combo9neg.addItem('session_4088341904')
        self.combo9neg.addItem('session_4322210317')
        self.combo9neg.currentIndexChanged.connect(self.indexChangedB)

        self.combo12pos = QComboBox(self)
        self.combo12pos.addItem('session_0166199610')
        self.combo12pos.addItem('session_7583047056')
        self.combo12pos.currentIndexChanged.connect(self.indexChangedB)

        self.combo12neg = QComboBox(self)
        self.combo12neg.addItem('session_0126772600')
        self.combo12neg.addItem('session_0170625567')
        self.combo12neg.addItem('session_0172860263')
        self.combo12neg.addItem('session_0172989910')
        self.combo12neg.currentIndexChanged.connect(self.indexChangedB)

        self.combo15pos = QComboBox(self)
        self.combo15pos.addItem('session_1567411744')
        self.combo15pos.addItem('session_1824070206')
        self.combo15pos.currentIndexChanged.connect(self.indexChangedB)

        self.combo15neg = QComboBox(self)
        self.combo15neg.addItem('session_0003960194')
        self.combo15neg.addItem('session_0128859274')
        self.combo15neg.currentIndexChanged.connect(self.indexChangedB)

        self.combo16pos = QComboBox(self)
        self.combo16pos.addItem('session_0005840196')
        self.combo16pos.addItem('session_0025450757')
        self.combo16pos.addItem('session_0083463746')
        self.combo16pos.addItem('session_0148970615')
        self.combo16pos.addItem('session_0155746039')
        self.combo16pos.currentIndexChanged.connect(self.indexChangedB)

        self.combo16neg = QComboBox(self)
        self.combo16neg.addItem('session_0064281061')
        self.combo16neg.addItem('session_4098547958')
        self.combo16neg.currentIndexChanged.connect(self.indexChangedB)

        self.combo20pos = QComboBox(self)
        self.combo20pos.addItem('session_0101735014')
        self.combo20pos.addItem('session_2532367006')
        self.combo20pos.currentIndexChanged.connect(self.indexChangedB)

        self.combo20neg = QComboBox(self)
        self.combo20neg.addItem('session_3379861047')
        self.combo20neg.addItem('session_3236365486')
        self.combo20neg.currentIndexChanged.connect(self.indexChangedB)

        self.combo21pos = QComboBox(self)
        self.combo21pos.addItem('session_2476629136')
        self.combo21pos.addItem('session_2681498481')
        self.combo21pos.currentIndexChanged.connect(self.indexChangedB)

        self.combo21neg = QComboBox(self)
        self.combo21neg.addItem('session_0080153528')
        self.combo21neg.addItem('session_2383353199')
        self.combo21neg.currentIndexChanged.connect(self.indexChangedB)

        self.combo23pos = QComboBox(self)
        self.combo23pos.addItem('session_0071280153')
        self.combo23pos.addItem('session_0139259699')
        self.combo23pos.currentIndexChanged.connect(self.indexChangedB)

        self.combo23neg = QComboBox(self)
        self.combo23neg.addItem('session_3436410606')
        self.combo23neg.addItem('session_3496233301')
        self.combo23neg.currentIndexChanged.connect(self.indexChangedB)

        self.combo29pos = QComboBox(self)
        self.combo29pos.addItem('session_2290361531')
        self.combo29pos.addItem('session_2540441733')
        self.combo29pos.currentIndexChanged.connect(self.indexChangedB)

        self.combo29neg = QComboBox(self)
        self.combo29neg.addItem('session_3147629890')
        self.combo29neg.addItem('session_3172392634')
        self.combo29neg.currentIndexChanged.connect(self.indexChangedB)

        self.combo35pos = QComboBox(self)
        self.combo35pos.addItem('session_0029922803')
        self.combo35pos.addItem('session_2110642502')
        self.combo35pos.currentIndexChanged.connect(self.indexChangedB)

        self.combo35neg = QComboBox(self)
        self.combo35neg.addItem('session_0111356050')
        self.combo35neg.addItem('session_2272584671')
        self.combo35neg.currentIndexChanged.connect(self.indexChangedB)

        self.combo_container_layout.addWidget(self.combo7pos)
        self.combo_container_layout.addWidget(self.combo7neg)
        self.combo_container_layout.addWidget(self.combo9pos)
        self.combo_container_layout.addWidget(self.combo9neg)
        self.combo_container_layout.addWidget(self.combo12pos)
        self.combo_container_layout.addWidget(self.combo12neg)
        self.combo_container_layout.addWidget(self.combo15pos)
        self.combo_container_layout.addWidget(self.combo15neg)
        self.combo_container_layout.addWidget(self.combo16pos)
        self.combo_container_layout.addWidget(self.combo16neg)
        self.combo_container_layout.addWidget(self.combo20pos)
        self.combo_container_layout.addWidget(self.combo20neg)
        self.combo_container_layout.addWidget(self.combo21pos)
        self.combo_container_layout.addWidget(self.combo21neg)
        self.combo_container_layout.addWidget(self.combo23pos)
        self.combo_container_layout.addWidget(self.combo23neg)
        self.combo_container_layout.addWidget(self.combo29pos)
        self.combo_container_layout.addWidget(self.combo29neg)
        self.combo_container_layout.addWidget(self.combo35pos)
        self.combo_container_layout.addWidget(self.combo35neg)

        self.combo_container.setLayout(self.combo_container_layout)

        form_layout = QFormLayout()

        form_layout.addRow(QLabel('Choose user:\t'), self.combo_user)
        form_layout.addRow(QLabel('Choose session type:\t'), self.combo_type)
        # the stacked layout is placed in place of the (meant to be) second combobox
        form_layout.addRow(QLabel('Choose session:\t'), self.combo_container)

        self.button_select = QPushButton('Show session')
        self.button_select.clicked.connect(self.draw_session)

        self.button_verify.clicked.connect(self.verify_session)

        form_layout.addRow(self.button_select)
        form_layout.addRow(self.button_verify)
        self.button_verify.setEnabled(False)

        self.setLayout(form_layout)

        # window settings
        self.resize(700, 900)
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        self.setWindowTitle('Balabit dataset')
        self.setWindowIcon(QIcon('mouse.ico'))
        self.show()

    def indexChangedA(self, index):
        self.select()

    def indexChangedB(self, index):
        self.select()

    def select(self):
        self.button_verify.setEnabled(False)
        self.text_threshold.hide()
        self.text_score.hide()
        self.label_cross_grey.hide()
        self.label_tick_grey.hide()
        self.label_tick.hide()
        self.label_cross.hide()
        self.label_threshold.hide()
        self.label_score.hide()
        self.label_image.hide()
        if self.combo_user.currentIndex() == 0 and self.combo_type.currentIndex() == 0:
            self.combo_container_layout.setCurrentIndex(0)
        if self.combo_user.currentIndex() == 0 and self.combo_type.currentIndex() == 1:
            self.combo_container_layout.setCurrentIndex(1)
        if self.combo_user.currentIndex() == 1 and self.combo_type.currentIndex() == 0:
            self.combo_container_layout.setCurrentIndex(2)
        if self.combo_user.currentIndex() == 1 and self.combo_type.currentIndex() == 1:
            self.combo_container_layout.setCurrentIndex(3)
        if self.combo_user.currentIndex() == 2 and self.combo_type.currentIndex() == 0:
            self.combo_container_layout.setCurrentIndex(4)
        if self.combo_user.currentIndex() == 2 and self.combo_type.currentIndex() == 1:
            self.combo_container_layout.setCurrentIndex(5)
        if self.combo_user.currentIndex() == 3 and self.combo_type.currentIndex() == 0:
            self.combo_container_layout.setCurrentIndex(6)
        if self.combo_user.currentIndex() == 3 and self.combo_type.currentIndex() == 1:
            self.combo_container_layout.setCurrentIndex(7)
        if self.combo_user.currentIndex() == 4 and self.combo_type.currentIndex() == 0:
            self.combo_container_layout.setCurrentIndex(8)
        if self.combo_user.currentIndex() == 4 and self.combo_type.currentIndex() == 1:
            self.combo_container_layout.setCurrentIndex(9)
        if self.combo_user.currentIndex() == 5 and self.combo_type.currentIndex() == 0:
            self.combo_container_layout.setCurrentIndex(10)
        if self.combo_user.currentIndex() == 5 and self.combo_type.currentIndex() == 1:
            self.combo_container_layout.setCurrentIndex(11)
        if self.combo_user.currentIndex() == 6 and self.combo_type.currentIndex() == 0:
            self.combo_container_layout.setCurrentIndex(12)
        if self.combo_user.currentIndex() == 6 and self.combo_type.currentIndex() == 1:
            self.combo_container_layout.setCurrentIndex(13)
        if self.combo_user.currentIndex() == 7 and self.combo_type.currentIndex() == 0:
            self.combo_container_layout.setCurrentIndex(14)
        if self.combo_user.currentIndex() == 7 and self.combo_type.currentIndex() == 1:
            self.combo_container_layout.setCurrentIndex(15)
        if self.combo_user.currentIndex() == 8 and self.combo_type.currentIndex() == 0:
            self.combo_container_layout.setCurrentIndex(16)
        if self.combo_user.currentIndex() == 8 and self.combo_type.currentIndex() == 1:
            self.combo_container_layout.setCurrentIndex(17)
        if self.combo_user.currentIndex() == 9 and self.combo_type.currentIndex() == 0:
            self.combo_container_layout.setCurrentIndex(18)
        if self.combo_user.currentIndex() == 9 and self.combo_type.currentIndex() == 1:
            self.combo_container_layout.setCurrentIndex(19)

    def draw_session(self):
        user = self.combo_user.itemText(self.combo_user.currentIndex())
        session = self.combo_container_layout.currentWidget(). \
            itemText(self.combo_container_layout.currentWidget().currentIndex())
        file_name = 'D:/Sapientia EMTE/final exam/softwares/MouseDynamics/test_files/' + user + '/' + session
        print(file_name)
        plotting_session.main(file_name)

        image_name = user + '_' + session + '.png'
        print(image_name)

        self.label_image.setPixmap(QPixmap(image_name))
        self.label_image.setGeometry(30, 180, 700, 470)
        self.label_image.show()
        self.button_verify.setEnabled(True)

    def verify_session(self):
        self.label_cross_grey.setPixmap(QPixmap('cross_grey.png'))
        self.label_cross_grey.setGeometry(450, 660, 100, 100)
        self.label_cross_grey.show()

        self.label_tick_grey.setPixmap(QPixmap('tick_grey.png'))
        self.label_tick_grey.setGeometry(450, 770, 100, 100)
        self.label_tick_grey.show()

        print('verify')

        user_7_thresh = 0.65
        user_9_thresh = 0.73
        user_12_thresh = 0.48
        user_15_thresh = 0.5
        user_16_thresh = 0.47
        user_20_thresh = 0.43
        user_21_thresh = 0.46
        user_23_thresh = 0.5
        user_29_thresh = 0.51
        user_35_thresh = 0.48

        user = self.combo_user.itemText(self.combo_user.currentIndex())
        session = self.combo_container_layout.currentWidget(). \
            itemText(self.combo_container_layout.currentWidget().currentIndex())
        score = verifying_session.main(user, session)
        print(score)

        if user == 'user7':
            threshold = user_7_thresh
        if user == 'user9':
            threshold = user_9_thresh
        if user == 'user12':
            threshold = user_12_thresh
        if user == 'user15':
            threshold = user_15_thresh
        if user == 'user16':
            threshold = user_16_thresh
        if user == 'user20':
            threshold = user_20_thresh
        if user == 'user21':
            threshold = user_21_thresh
        if user == 'user23':
            threshold = user_23_thresh
        if user == 'user29':
            threshold = user_29_thresh
        if user == 'user35':
            threshold = user_35_thresh

        self.label_threshold.setText('THRESHOLD:')
        self.label_threshold.move(130, 730)
        self.label_threshold.show()

        self.text_threshold.setText(str(threshold))
        self.text_threshold.move(250, 730)
        #self.text_threshold.setEnabled(False)
        self.text_threshold.show()

        self.label_score.setText('SCORE:')
        self.label_score.move(130, 770)
        self.label_score.show()

        self.text_score.setText(str(round(score, 2)))
        self.text_score.move(250, 770)
        #self.text_score.setEnabled(False)
        self.text_score.show()

        if score < threshold:
            self.label_cross.setPixmap(QPixmap('cross.png'))
            self.label_cross.setGeometry(450, 660, 100, 100)
            self.label_cross.show()
        else:

            self.label_tick.setPixmap(QPixmap('tick.png'))
            self.label_tick.setGeometry(450, 770, 100, 100)
            self.label_tick.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    userInterface = UserInterface()
    sys.exit(app.exec_())

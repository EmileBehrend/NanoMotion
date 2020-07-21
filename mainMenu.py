# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainMenu.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1121, 1016)
        MainWindow.setAcceptDrops(True)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.mplwindow = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mplwindow.sizePolicy().hasHeightForWidth())
        self.mplwindow.setSizePolicy(sizePolicy)
        self.mplwindow.setAcceptDrops(True)
        self.mplwindow.setObjectName("mplwindow")
        self.mplvl = QtWidgets.QVBoxLayout(self.mplwindow)
        self.mplvl.setObjectName("mplvl")
        self.horizontalLayout.addWidget(self.mplwindow)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label_pix_size = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_pix_size.sizePolicy().hasHeightForWidth())
        self.label_pix_size.setSizePolicy(sizePolicy)
        self.label_pix_size.setMaximumSize(QtCore.QSize(170, 16777215))
        self.label_pix_size.setObjectName("label_pix_size")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_pix_size)
        self.lineEdit_pix_size = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_pix_size.sizePolicy().hasHeightForWidth())
        self.lineEdit_pix_size.setSizePolicy(sizePolicy)
        self.lineEdit_pix_size.setMaximumSize(QtCore.QSize(50, 16777215))
        self.lineEdit_pix_size.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.lineEdit_pix_size.setObjectName("lineEdit_pix_size")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_pix_size)
        self.label_magn = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_magn.sizePolicy().hasHeightForWidth())
        self.label_magn.setSizePolicy(sizePolicy)
        self.label_magn.setMaximumSize(QtCore.QSize(170, 16777215))
        self.label_magn.setObjectName("label_magn")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_magn)
        self.lineEdit_magn = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_magn.sizePolicy().hasHeightForWidth())
        self.lineEdit_magn.setSizePolicy(sizePolicy)
        self.lineEdit_magn.setMaximumSize(QtCore.QSize(50, 16777215))
        self.lineEdit_magn.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.lineEdit_magn.setObjectName("lineEdit_magn")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineEdit_magn)
        self.label_sub_pix = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_sub_pix.sizePolicy().hasHeightForWidth())
        self.label_sub_pix.setSizePolicy(sizePolicy)
        self.label_sub_pix.setMaximumSize(QtCore.QSize(170, 16777215))
        self.label_sub_pix.setObjectName("label_sub_pix")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_sub_pix)
        self.lineEdit_sub_pix = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_sub_pix.sizePolicy().hasHeightForWidth())
        self.lineEdit_sub_pix.setSizePolicy(sizePolicy)
        self.lineEdit_sub_pix.setMaximumSize(QtCore.QSize(50, 16777215))
        self.lineEdit_sub_pix.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.lineEdit_sub_pix.setInputMask("")
        self.lineEdit_sub_pix.setObjectName("lineEdit_sub_pix")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEdit_sub_pix)
        self.label_fps = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_fps.sizePolicy().hasHeightForWidth())
        self.label_fps.setSizePolicy(sizePolicy)
        self.label_fps.setMaximumSize(QtCore.QSize(170, 16777215))
        self.label_fps.setObjectName("label_fps")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_fps)
        self.lineEdit_fps = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_fps.sizePolicy().hasHeightForWidth())
        self.lineEdit_fps.setSizePolicy(sizePolicy)
        self.lineEdit_fps.setMaximumSize(QtCore.QSize(50, 16777215))
        self.lineEdit_fps.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.lineEdit_fps.setObjectName("lineEdit_fps")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.lineEdit_fps)
        self.label_start_frame = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_start_frame.sizePolicy().hasHeightForWidth())
        self.label_start_frame.setSizePolicy(sizePolicy)
        self.label_start_frame.setMaximumSize(QtCore.QSize(170, 16777215))
        self.label_start_frame.setObjectName("label_start_frame")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_start_frame)
        self.lineEdit_start_frame = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_start_frame.sizePolicy().hasHeightForWidth())
        self.lineEdit_start_frame.setSizePolicy(sizePolicy)
        self.lineEdit_start_frame.setMaximumSize(QtCore.QSize(50, 16777215))
        self.lineEdit_start_frame.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.lineEdit_start_frame.setObjectName("lineEdit_start_frame")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.lineEdit_start_frame)
        self.label_stop_frame = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_stop_frame.sizePolicy().hasHeightForWidth())
        self.label_stop_frame.setSizePolicy(sizePolicy)
        self.label_stop_frame.setMaximumSize(QtCore.QSize(170, 16777215))
        self.label_stop_frame.setObjectName("label_stop_frame")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_stop_frame)
        self.lineEdit_stop_frame = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_stop_frame.sizePolicy().hasHeightForWidth())
        self.lineEdit_stop_frame.setSizePolicy(sizePolicy)
        self.lineEdit_stop_frame.setMaximumSize(QtCore.QSize(50, 16777215))
        self.lineEdit_stop_frame.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.lineEdit_stop_frame.setObjectName("lineEdit_stop_frame")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.lineEdit_stop_frame)
        self.label_w = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_w.sizePolicy().hasHeightForWidth())
        self.label_w.setSizePolicy(sizePolicy)
        self.label_w.setMaximumSize(QtCore.QSize(170, 16777215))
        self.label_w.setObjectName("label_w")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_w)
        self.lineEdit_w = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_w.sizePolicy().hasHeightForWidth())
        self.lineEdit_w.setSizePolicy(sizePolicy)
        self.lineEdit_w.setMaximumSize(QtCore.QSize(50, 16777215))
        self.lineEdit_w.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.lineEdit_w.setObjectName("lineEdit_w")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.lineEdit_w)
        self.label_h = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_h.sizePolicy().hasHeightForWidth())
        self.label_h.setSizePolicy(sizePolicy)
        self.label_h.setMaximumSize(QtCore.QSize(170, 16777215))
        self.label_h.setObjectName("label_h")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.LabelRole, self.label_h)
        self.lineEdit_h = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_h.sizePolicy().hasHeightForWidth())
        self.lineEdit_h.setSizePolicy(sizePolicy)
        self.lineEdit_h.setMaximumSize(QtCore.QSize(50, 16777215))
        self.lineEdit_h.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.lineEdit_h.setObjectName("lineEdit_h")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.FieldRole, self.lineEdit_h)
        self.checkBox_substract = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBox_substract.sizePolicy().hasHeightForWidth())
        self.checkBox_substract.setSizePolicy(sizePolicy)
        self.checkBox_substract.setMaximumSize(QtCore.QSize(170, 16777215))
        self.checkBox_substract.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.checkBox_substract.setObjectName("checkBox_substract")
        self.formLayout.setWidget(9, QtWidgets.QFormLayout.LabelRole, self.checkBox_substract)
        self.lineEdit_substract_lvl = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_substract_lvl.sizePolicy().hasHeightForWidth())
        self.lineEdit_substract_lvl.setSizePolicy(sizePolicy)
        self.lineEdit_substract_lvl.setMaximumSize(QtCore.QSize(50, 16777215))
        self.lineEdit_substract_lvl.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.lineEdit_substract_lvl.setObjectName("lineEdit_substract_lvl")
        self.formLayout.setWidget(9, QtWidgets.QFormLayout.FieldRole, self.lineEdit_substract_lvl)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.formLayout.setWidget(10, QtWidgets.QFormLayout.LabelRole, self.label)
        self.comboBox_substract_col = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_substract_col.sizePolicy().hasHeightForWidth())
        self.comboBox_substract_col.setSizePolicy(sizePolicy)
        self.comboBox_substract_col.setMaximumSize(QtCore.QSize(70, 16777215))
        self.comboBox_substract_col.setObjectName("comboBox_substract_col")
        self.formLayout.setWidget(10, QtWidgets.QFormLayout.FieldRole, self.comboBox_substract_col)
        self.lineEdit_chop_sec = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_chop_sec.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_chop_sec.sizePolicy().hasHeightForWidth())
        self.lineEdit_chop_sec.setSizePolicy(sizePolicy)
        self.lineEdit_chop_sec.setMaximumSize(QtCore.QSize(50, 16777215))
        self.lineEdit_chop_sec.setStatusTip("")
        self.lineEdit_chop_sec.setObjectName("lineEdit_chop_sec")
        self.formLayout.setWidget(11, QtWidgets.QFormLayout.FieldRole, self.lineEdit_chop_sec)
        self.label_chop_sec = QtWidgets.QLabel(self.centralwidget)
        self.label_chop_sec.setEnabled(False)
        self.label_chop_sec.setObjectName("label_chop_sec")
        self.formLayout.setWidget(11, QtWidgets.QFormLayout.LabelRole, self.label_chop_sec)
        self.checkBox_track = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_track.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.checkBox_track.setChecked(True)
        self.checkBox_track.setObjectName("checkBox_track")
        self.formLayout.setWidget(12, QtWidgets.QFormLayout.LabelRole, self.checkBox_track)
        self.checkBox_compare_first = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_compare_first.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.checkBox_compare_first.setObjectName("checkBox_compare_first")
        self.formLayout.setWidget(13, QtWidgets.QFormLayout.LabelRole, self.checkBox_compare_first)
        self.checkBox_live_preview = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_live_preview.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.checkBox_live_preview.setObjectName("checkBox_live_preview")
        self.formLayout.setWidget(14, QtWidgets.QFormLayout.LabelRole, self.checkBox_live_preview)
        self.verticalLayout.addLayout(self.formLayout)
        self.views = QtWidgets.QListWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.views.sizePolicy().hasHeightForWidth())
        self.views.setSizePolicy(sizePolicy)
        self.views.setMaximumSize(QtCore.QSize(200, 16777215))
        self.views.setObjectName("views")
        self.verticalLayout.addWidget(self.views)
        self.boxes = QtWidgets.QListWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.boxes.sizePolicy().hasHeightForWidth())
        self.boxes.setSizePolicy(sizePolicy)
        self.boxes.setMaximumSize(QtCore.QSize(200, 16777215))
        self.boxes.setObjectName("boxes")
        self.verticalLayout.addWidget(self.boxes)
        self.horizontalLayout.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1121, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuView_plot = QtWidgets.QMenu(self.menuView)
        self.menuView_plot.setObjectName("menuView_plot")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionConvert = QtWidgets.QAction(MainWindow)
        self.actionConvert.setObjectName("actionConvert")
        self.actionSubstract = QtWidgets.QAction(MainWindow)
        self.actionSubstract.setObjectName("actionSubstract")
        self.actionPixel_size = QtWidgets.QAction(MainWindow)
        self.actionPixel_size.setObjectName("actionPixel_size")
        self.actionMagnification = QtWidgets.QAction(MainWindow)
        self.actionMagnification.setObjectName("actionMagnification")
        self.actionSub_pixel_level = QtWidgets.QAction(MainWindow)
        self.actionSub_pixel_level.setObjectName("actionSub_pixel_level")
        self.actionEdit_objects_list = QtWidgets.QAction(MainWindow)
        self.actionEdit_objects_list.setObjectName("actionEdit_objects_list")
        self.actionViolin = QtWidgets.QAction(MainWindow)
        self.actionViolin.setCheckable(True)
        self.actionViolin.setChecked(False)
        self.actionViolin.setObjectName("actionViolin")
        self.actionPos = QtWidgets.QAction(MainWindow)
        self.actionPos.setCheckable(True)
        self.actionPos.setObjectName("actionPos")
        self.actionSub_frameset = QtWidgets.QAction(MainWindow)
        self.actionSub_frameset.setObjectName("actionSub_frameset")
        self.actionAdd_box = QtWidgets.QAction(MainWindow)
        self.actionAdd_box.setObjectName("actionAdd_box")
        self.actionStart_analysis = QtWidgets.QAction(MainWindow)
        self.actionStart_analysis.setObjectName("actionStart_analysis")
        self.actionx_shift = QtWidgets.QAction(MainWindow)
        self.actionx_shift.setCheckable(True)
        self.actionx_shift.setObjectName("actionx_shift")
        self.actiony_shift = QtWidgets.QAction(MainWindow)
        self.actiony_shift.setCheckable(True)
        self.actiony_shift.setObjectName("actiony_shift")
        self.actionShow_results = QtWidgets.QAction(MainWindow)
        self.actionShow_results.setObjectName("actionShow_results")
        self.actionExport_results = QtWidgets.QAction(MainWindow)
        self.actionExport_results.setObjectName("actionExport_results")
        self.actionViolin_all_on_one = QtWidgets.QAction(MainWindow)
        self.actionViolin_all_on_one.setCheckable(True)
        self.actionViolin_all_on_one.setChecked(True)
        self.actionViolin_all_on_one.setObjectName("actionViolin_all_on_one")
        self.actionViolin_chop = QtWidgets.QAction(MainWindow)
        self.actionViolin_chop.setCheckable(True)
        self.actionViolin_chop.setObjectName("actionViolin_chop")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionExport_results)
        self.menuView_plot.addAction(self.actionViolin)
        self.menuView_plot.addAction(self.actionViolin_all_on_one)
        self.menuView_plot.addAction(self.actionPos)
        self.menuView_plot.addAction(self.actionx_shift)
        self.menuView_plot.addAction(self.actiony_shift)
        self.menuView_plot.addAction(self.actionViolin_chop)
        self.menuView.addAction(self.actionSubstract)
        self.menuView.addAction(self.menuView_plot.menuAction())
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "LPVM-NanoMotionAnalysis"))
        self.label_pix_size.setText(_translate("MainWindow", "Pixel size μm"))
        self.lineEdit_pix_size.setToolTip(_translate("MainWindow", "Sensor pixel size"))
        self.lineEdit_pix_size.setText(_translate("MainWindow", "1.12"))
        self.label_magn.setText(_translate("MainWindow", "Magnification"))
        self.lineEdit_magn.setToolTip(_translate("MainWindow", "Enlarging factor"))
        self.lineEdit_magn.setText(_translate("MainWindow", "1"))
        self.label_sub_pix.setText(_translate("MainWindow", "Sub-pixel level"))
        self.lineEdit_sub_pix.setToolTip(_translate("MainWindow", "100 is usually good"))
        self.lineEdit_sub_pix.setText(_translate("MainWindow", "100"))
        self.label_fps.setText(_translate("MainWindow", "fps"))
        self.lineEdit_fps.setText(_translate("MainWindow", "25"))
        self.label_start_frame.setText(_translate("MainWindow", "Start frame"))
        self.lineEdit_start_frame.setText(_translate("MainWindow", "0"))
        self.label_stop_frame.setText(_translate("MainWindow", "Stop frame"))
        self.lineEdit_stop_frame.setText(_translate("MainWindow", "199"))
        self.label_w.setText(_translate("MainWindow", "Box width"))
        self.lineEdit_w.setText(_translate("MainWindow", "40"))
        self.label_h.setText(_translate("MainWindow", "Box height"))
        self.lineEdit_h.setText(_translate("MainWindow", "40"))
        self.checkBox_substract.setText(_translate("MainWindow", "Substract, nframes"))
        self.lineEdit_substract_lvl.setToolTip(_translate("MainWindow", "How many times in range to do this ?"))
        self.lineEdit_substract_lvl.setText(_translate("MainWindow", "5"))
        self.label.setText(_translate("MainWindow", "Substract color scale"))
        self.lineEdit_chop_sec.setToolTip(_translate("MainWindow", "Numbers of second of every slice"))
        self.label_chop_sec.setText(_translate("MainWindow", "Chop duration seconds"))
        self.checkBox_track.setText(_translate("MainWindow", "Track particle "))
        self.checkBox_compare_first.setText(_translate("MainWindow", "Compare to 1st img "))
        self.checkBox_live_preview.setText(_translate("MainWindow", "Live preview"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.menuView_plot.setTitle(_translate("MainWindow", "View plot"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionOpen.setStatusTip(_translate("MainWindow", "Open a file supported by PIMS (http://soft-matter.github.io/pims)"))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionConvert.setText(_translate("MainWindow", "Convert"))
        self.actionConvert.setStatusTip(_translate("MainWindow", "Convert the previously opened file to .mp4 using ffmpeg"))
        self.actionSubstract.setText(_translate("MainWindow", "Substract"))
        self.actionSubstract.setStatusTip(_translate("MainWindow", "Substract the last frame from the 1st frame"))
        self.actionPixel_size.setText(_translate("MainWindow", "Pixel size"))
        self.actionMagnification.setText(_translate("MainWindow", "Magnification"))
        self.actionMagnification.setStatusTip(_translate("MainWindow", "Defines the optical zoom"))
        self.actionSub_pixel_level.setText(_translate("MainWindow", "Sub pixel level"))
        self.actionSub_pixel_level.setStatusTip(_translate("MainWindow", "100 is usually good"))
        self.actionEdit_objects_list.setText(_translate("MainWindow", "Edit objects list"))
        self.actionViolin.setText(_translate("MainWindow", "Violin_step_length"))
        self.actionPos.setText(_translate("MainWindow", "pos(t)"))
        self.actionSub_frameset.setText(_translate("MainWindow", "Sub-frameset"))
        self.actionSub_frameset.setStatusTip(_translate("MainWindow", "Select a set to show and analyse"))
        self.actionAdd_box.setText(_translate("MainWindow", "Add box"))
        self.actionAdd_box.setStatusTip(_translate("MainWindow", "Add zone to analyse. Modify it on screen"))
        self.actionAdd_box.setShortcut(_translate("MainWindow", "A"))
        self.actionStart_analysis.setText(_translate("MainWindow", "Start analysis"))
        self.actionStart_analysis.setShortcut(_translate("MainWindow", "S"))
        self.actionx_shift.setText(_translate("MainWindow", "x(t)_shift"))
        self.actiony_shift.setText(_translate("MainWindow", "y(t)_shift"))
        self.actionShow_results.setText(_translate("MainWindow", "Show results"))
        self.actionExport_results.setText(_translate("MainWindow", "Export results"))
        self.actionViolin_all_on_one.setText(_translate("MainWindow", "Violin_all_on_one"))
        self.actionViolin_chop.setText(_translate("MainWindow", "Violin_chop"))

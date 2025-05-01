import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import face_recognition
import os
import requests
from bs4 import BeautifulSoup
import threading
import webbrowser
from datetime import datetime

class ResponsiveFaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("نظام التعرف الذكي على الوجوه - الإصدار المتجاوب")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f2f5")
        self.root.minsize(800, 600)
        
        # متغيرات النظام
        self.recognized_name = None
        self.cap = None
        self.is_camera_on = False
        self.known_face_encodings = []
        self.known_face_names = []
        self.current_frame = None
        
        # إعدادات التجاوب
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # تحميل الأصول البصرية
        self.load_assets()
        
        # إنشاء الواجهة
        self.setup_responsive_ui()
        
        # تحميل الوجوه المعروفة
        self.load_known_faces()
        
        # ربط أحداث تغيير الحجم
        self.root.bind("<Configure>", self.on_window_resize)

    def load_assets(self):
        """تحميل الأصول مع التعامل مع الأخطاء"""
        try:
            # أيقونات افتراضية باستخدام رمز نصي
            self.icons = {
                'camera': self.create_text_icon("📷"),
                'search': self.create_text_icon("🔍"),
                'add': self.create_text_icon("➕"),
                'exit': self.create_text_icon("🚪"),
                'logo': self.create_text_icon("👤", size=40)
            }
        except Exception as e:
            print(f"تحذير: {e}")
            self.icons = {}

    def create_text_icon(self, text, size=30):
        """إنشاء أيقونة من نص"""
        img = Image.new('RGB', (size, size), color='#3498db')
        return ImageTk.PhotoImage(img)

    def setup_responsive_ui(self):
        """إنشاء واجهة متجاوبة"""
        # شريط العنوان
        self.header = tk.Frame(self.root, bg="#1a5276", height=80)
        self.header.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.header.grid_columnconfigure(0, weight=1)
        
        # لوحة العنوان
        title_container = tk.Frame(self.header, bg="#1a5276")
        title_container.grid(row=0, column=0, sticky="e", padx=20)
        
        if 'logo' in self.icons:
            logo_label = tk.Label(title_container, image=self.icons['logo'], bg="#1a5276")
            logo_label.grid(row=0, column=1, padx=10)
        
        title_label = tk.Label(title_container, 
                             text="نظام التعرف الذكي على الوجوه", 
                             font=("Arial", 20, "bold"), 
                             fg="white", bg="#1a5276")
        title_label.grid(row=0, column=0, padx=10)
        
        # المنطقة الرئيسية
        self.main_container = tk.Frame(self.root, bg="#f0f2f5")
        self.main_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0,10))
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=3)
        self.main_container.grid_columnconfigure(1, weight=1)
        
        # لوحة العرض
        self.display_panel = tk.Frame(self.main_container, bg="#ffffff", 
                                    bd=2, relief=tk.RAISED)
        self.display_panel.grid(row=0, column=0, sticky="nsew", padx=(0,10))
        self.display_panel.grid_rowconfigure(1, weight=1)
        self.display_panel.grid_columnconfigure(0, weight=1)
        
        # شاشة الكاميرا
        self.camera_frame = tk.Frame(self.display_panel, bg="#000000")
        self.camera_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        self.camera_label = tk.Label(self.camera_frame, bg="black")
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # لوحة معلومات الكاميرا
        self.camera_info = tk.Frame(self.display_panel, bg="#ffffff")
        self.camera_info.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        self.status_label = tk.Label(self.camera_info, text="الحالة: جاهز", 
                                   font=("Arial", 11), fg="#333333", bg="#ffffff")
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        self.recognition_label = tk.Label(self.camera_info, 
                                        text="التعرف: غير معروف", 
                                        font=("Arial", 11), 
                                        fg="#e67e22", bg="#ffffff")
        self.recognition_label.pack(side=tk.RIGHT, padx=10)
        
        # لوحة التحكم
        self.control_panel = tk.Frame(self.main_container, bg="#ffffff", 
                                     bd=2, relief=tk.RAISED)
        self.control_panel.grid(row=0, column=1, sticky="nsew")
        
        # أزرار التحكم
        btn_style = {
            "font": ("Arial", 11), 
            "bg": "#3498db", 
            "fg": "white", 
            "bd": 0, 
            "padx": 10, 
            "pady": 8,
            "width": 15
        }
        
        self.camera_btn = tk.Button(self.control_panel, 
                                  text="تشغيل الكاميرا", 
                                  command=self.toggle_camera,
                                  **btn_style)
        self.camera_btn.grid(row=0, column=0, pady=10, padx=10, sticky="ew")
        
        self.search_btn = tk.Button(self.control_panel, 
                                  text="البحث في ويكيبيديا", 
                                  command=self.search_wikipedia,
                                  state=tk.DISABLED,
                                  **btn_style)
        self.search_btn.grid(row=1, column=0, pady=10, padx=10, sticky="ew")
        
        self.add_face_btn = tk.Button(self.control_panel, 
                                    text="إضافة وجه جديد", 
                                    command=self.add_new_face,
                                    **btn_style)
        self.add_face_btn.grid(row=2, column=0, pady=10, padx=10, sticky="ew")
        
        self.exit_btn = tk.Button(self.control_panel, 
                                text="خروج", 
                                command=self.exit_app,
                                **btn_style)
        self.exit_btn.grid(row=3, column=0, pady=10, padx=10, sticky="ew")
        
        # لوحة المعلومات
        self.info_panel = tk.Frame(self.root, bg="#ffffff", 
                                 bd=2, relief=tk.RAISED)
        self.info_panel.grid(row=2, column=0, sticky="ew", padx=10, pady=(0,10))
        
        info_title = tk.Label(self.info_panel, 
                            text="معلومات من ويكيبيديا", 
                            font=("Arial", 12, "bold"), 
                            fg="white", bg="#1a5276")
        info_title.pack(fill=tk.X, pady=(5,10))
        
        self.wiki_text = scrolledtext.ScrolledText(self.info_panel, 
                                                 wrap=tk.WORD, 
                                                 font=("Arial", 10), 
                                                 bg="#f9f9f9", fg="#333333",
                                                 padx=10, pady=10)
        self.wiki_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0,5))
        self.wiki_text.config(state=tk.DISABLED)
        
        # شريط الحالة
        self.status_bar = tk.Frame(self.root, bg="#1a5276", height=25)
        self.status_bar.grid(row=3, column=0, sticky="ew", padx=10, pady=(0,10))
        
        self.time_label = tk.Label(self.status_bar, 
                                 text="", 
                                 font=("Arial", 9), 
                                 fg="white", bg="#1a5276")
        self.time_label.pack(side=tk.RIGHT, padx=10)
        
        self.update_clock()

    def on_window_resize(self, event):
        """تعديل العناصر عند تغيير حجم النافذة"""
        width = self.root.winfo_width()
        
        # تعديل حجم الخط حسب حجم النافذة
        if width < 1000:
            new_font = ("Arial", 9)
        elif width < 1200:
            new_font = ("Arial", 10)
        else:
            new_font = ("Arial", 11)
            
        # تطبيق التغييرات على العناصر
        for widget in [self.status_label, self.recognition_label, self.time_label]:
            widget.config(font=new_font)
            
        for btn in [self.camera_btn, self.search_btn, self.add_face_btn, self.exit_btn]:
            btn.config(font=new_font)

    def update_clock(self):
        """تحديث الوقت في شريط الحالة"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=f"التاريخ والوقت: {now}")
        self.root.after(1000, self.update_clock)

    def load_known_faces(self):
        """تحميل الوجوه المعروفة"""
        path = 'persons'
        if not os.path.exists(path):
            os.makedirs(path)
            self.status_label.config(text="الحالة: مجلد persons تم إنشاؤه")
            return
        
        images = []
        classNames = []
        personsList = os.listdir(path)
        
        for cl in personsList:
            curImg = cv2.imread(f'{path}/{cl}')
            if curImg is not None:
                images.append(curImg)
                classNames.append(os.path.splitext(cl)[0])
        
        if images:
            self.status_label.config(text=f"الحالة: تم تحميل {len(images)} وجوه معروفة")
            self.known_face_names = classNames
            threading.Thread(target=self.encode_faces, args=(images,), daemon=True).start()
        else:
            self.status_label.config(text="الحالة: لم يتم العثور على وجوه معروفة")

    def encode_faces(self, images):
        """تشفير الوجوه"""
        encodeList = []
        for i, img in enumerate(images):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
            
            self.status_label.config(text=f"الحالة: جارٍ تشفير الوجوه ({i+1}/{len(images)})")
        
        self.known_face_encodings = encodeList
        self.status_label.config(text=f"الحالة: جاهز - {len(encodeList)} وجوه مشفرة")
        messagebox.showinfo("تم", "تم تحميل وتشفير الوجوه المعروفة بنجاح")

    def toggle_camera(self):
        """تشغيل/إيقاف الكاميرا"""
        if not self.is_camera_on:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """تشغيل الكاميرا"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("خطأ", "تعذر فتح الكاميرا")
            return
        
        self.is_camera_on = True
        self.camera_btn.config(text="إيقاف الكاميرا", bg="#e74c3c")
        self.search_btn.config(state=tk.DISABLED)
        self.status_label.config(text="الحالة: الكاميرا تعمل - جاري البحث عن وجوه...")
        self.update_camera()

    def stop_camera(self):
        """إيقاف الكاميرا"""
        if self.cap:
            self.is_camera_on = False
            self.cap.release()
            self.camera_btn.config(text="تشغيل الكاميرا", bg="#3498db")
            self.search_btn.config(state=tk.NORMAL if self.recognized_name else tk.DISABLED)
            self.status_label.config(text="الحالة: جاهز")
            self.camera_label.config(image='', bg='black')

    def update_camera(self):
        """تحديث إطار الكاميرا"""
        if self.is_camera_on:
            success, img = self.cap.read()
            if success:
                # معالجة الصورة للتعرف على الوجوه
                imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
                
                faceLocations = face_recognition.face_locations(imgS)
                encodesCurFrame = face_recognition.face_encodings(imgS, faceLocations)
                
                for encodeFace, faceLoc in zip(encodesCurFrame, faceLocations):
                    matches = face_recognition.compare_faces(self.known_face_encodings, encodeFace)
                    faceDis = face_recognition.face_distance(self.known_face_encodings, encodeFace)
                    matchIndex = np.argmin(faceDis)
                    
                    if matches[matchIndex]:
                        name = self.known_face_names[matchIndex].title()
                        self.recognized_name = name
                        self.recognition_label.config(text=f"التعرف: {name}", fg="#27ae60")
                        
                        # رسم مربع حول الوجه المعروف
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, name, (x1 + 6, y2 - 6),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                        
                        self.stop_camera()
                        self.search_btn.config(state=tk.NORMAL)
                        break
                
                # عرض الصورة في الواجهة
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                
                # ضبط حجم الصورة حسب حجم الإطار
                frame_width = self.camera_frame.winfo_width()
                frame_height = self.camera_frame.winfo_height()
                
                if frame_width > 0 and frame_height > 0:
                    img.thumbnail((frame_width, frame_height))
                
                img = ImageTk.PhotoImage(image=img)
                
                self.camera_label.img = img
                self.camera_label.config(image=img)
            
            self.root.after(10, self.update_camera)

    def search_wikipedia(self):
        """البحث في ويكيبيديا"""
        if not self.recognized_name:
            messagebox.showwarning("تحذير", "لم يتم التعرف على أي وجه بعد")
            return
        
        self.status_label.config(text=f"الحالة: جاري البحث عن {self.recognized_name} في ويكيبيديا...")
        self.wiki_text.config(state=tk.NORMAL)
        self.wiki_text.delete(1.0, tk.END)
        self.wiki_text.insert(tk.END, f"جارٍ جمع المعلومات عن {self.recognized_name}...\n")
        self.wiki_text.config(state=tk.DISABLED)
        
        threading.Thread(target=self._perform_wikipedia_search, daemon=True).start()

    def _perform_wikipedia_search(self):
        """تنفيذ البحث في ويكيبيديا"""
        name = self.recognized_name
        wiki_name = name.replace(' ', '_')
        url = f"https://en.wikipedia.org/wiki/{wiki_name}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            self._update_wiki_text(f"❌ خطأ أثناء جلب الصفحة: {e}\n")
            return
        
        soup = BeautifulSoup(response.text, 'html.parser')
        infobox = soup.find('table', {'class': 'infobox'})
        
        if not infobox:
            self._update_wiki_text("❌ لم يتم العثور على معلومات.\n")
            return
        
        # جمع المعلومات
        info_text = f"معلومات عن {name} من ويكيبيديا:\n\n"
        for row in infobox.find_all('tr'):
            header = row.find('th')
            data = row.find('td')
            if header and data:
                key = header.text.strip()
                value = data.text.strip().replace('\n', ' ')
                info_text += f"• {key}: {value}\n"
        
        info_text += f"\nرابط المقالة: {url}"
        self._update_wiki_text(info_text)
        self.status_label.config(text=f"الحالة: تم تحميل معلومات {name} من ويكيبيديا")

    def _update_wiki_text(self, text):
        self.wiki_text.config(state=tk.NORMAL)
        self.wiki_text.delete(1.0, tk.END)
        self.wiki_text.insert(tk.END, text)
        self.wiki_text.config(state=tk.DISABLED)
        
        self.wiki_text.tag_config("link", foreground="#3498db", underline=1)
        self.wiki_text.tag_add("link", "end-2l", "end")
        self.wiki_text.tag_bind("link", "<Button-1>", 
                              lambda e: webbrowser.open(f"https://en.wikipedia.org/wiki/{self.recognized_name.replace(' ', '_')}"))

    def add_new_face(self):
        """إضافة وجه جديد"""
        if self.is_camera_on:
            messagebox.showwarning("تحذير", "يجب إيقاف الكاميرا أولاً")
            return
        
        file_path = filedialog.askopenfilename(
            title="اختر صورة للوجه الجديد",
            filetypes=[("ملفات الصور", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return
        
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("خطأ", "تعذر تحميل الصورة المحددة")
            return
        
        name = simpledialog.askstring("إدخال الاسم", "أدخل اسم الشخص:")
        if not name:
            return
        
        save_path = f"persons/{name}.jpg"
        cv2.imwrite(save_path, img)
        
        self.load_known_faces()
        messagebox.showinfo("تم", f"تمت إضافة {name} إلى النظام بنجاح")

    def exit_app(self):
        if self.is_camera_on:
            self.stop_camera()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ResponsiveFaceRecognitionApp(root)
    root.mainloop()
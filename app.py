import tkinter as tk

def change_label():
    label.config(text="Bạn đã nhấn nút!")

# Tạo cửa sổ chính
window = tk.Tk()
window.title("Chương trình Tkinter đơn giản")
window.geometry("300x200")

# Tạo nhãn
label = tk.Label(window, text="Chào mừng đến với Tkinter!", font=("Arial", 14))
label.pack(pady=20)

# Tạo nút
button = tk.Button(window, text="Nhấn tôi", command=change_label)
button.pack()

# Chạy vòng lặp chính
window.mainloop()
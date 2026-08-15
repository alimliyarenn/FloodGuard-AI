import os
import glob
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
import matplotlib.pyplot as plt
import requests

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score


# =========================================================
# DOSYA OKUMA
# =========================================================
def csv_dosyasi_bul():
    olasi_dosyalar = [
        "sel_verileri.csv",
        "sel_verileri(3).csv",
        "sel_verileri (3).csv"
    ]

    for dosya in olasi_dosyalar:
        if os.path.exists(dosya):
            return dosya

    bulunanlar = glob.glob("*.csv")
    if bulunanlar:
        return bulunanlar[0]

    return None


csv_yolu = csv_dosyasi_bul()

if csv_yolu is None:
    messagebox.showerror(
        "Hata",
        "CSV dosyası bulunamadı.\nLütfen sel_verileri.csv dosyasını kodla aynı klasöre koyunuz."
    )
    raise SystemExit

try:
    df = pd.read_csv(csv_yolu)
except Exception as hata:
    messagebox.showerror("Hata", f"CSV okunamadı:\n{hata}")
    raise SystemExit


# =========================================================
# VERİ VE MODEL HAZIRLIĞI
# =========================================================
ozellikler = [
    "yagis_miktari",
    "dere_seviyesi",
    "zemin_egimi",
    "altyapi_durumu",
    "gecmis_sel_sayisi",
    "nufus_yogunlugu"
]

zorunlu_sutunlar = ozellikler + ["risk_sinifi", "bolge", "enlem", "boylam"]
eksik_sutunlar = [s for s in zorunlu_sutunlar if s not in df.columns]

if eksik_sutunlar:
    messagebox.showerror(
        "Hata",
        "CSV içinde eksik sütunlar var:\n" + ", ".join(eksik_sutunlar)
    )
    raise SystemExit

X = df[ozellikler]
y = df["risk_sinifi"]

sinif_modeli = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
sinif_modeli.fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y if y.value_counts().min() >= 2 else None
)

test_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
test_model.fit(X_train, y_train)

test_tahmin = test_model.predict(X_test)

dogruluk = accuracy_score(y_test, test_tahmin)
f1 = f1_score(y_test, test_tahmin, average="weighted")

risk_puan_map = {
    "Dusuk": 25,
    "Orta": 50,
    "Yuksek": 75,
    "Kritik": 95
}

df["risk_puani"] = df["risk_sinifi"].map(risk_puan_map)

regresyon_modeli = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
regresyon_modeli.fit(X, df["risk_puani"])

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X,
    df["risk_puani"],
    test_size=0.25,
    random_state=42
)

reg_test_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
reg_test_model.fit(Xr_train, yr_train)

reg_tahmin = reg_test_model.predict(Xr_test)
mae = mean_absolute_error(yr_test, reg_tahmin)

df["model_tahmini"] = sinif_modeli.predict(X)
df["model_risk_puani"] = regresyon_modeli.predict(X).round(1)


# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================
def guncel_hava_cek(enlem, boylam):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={enlem}&longitude={boylam}"
        f"&current=temperature_2m,relative_humidity_2m,rain"
        f"&hourly=rain"
        f"&forecast_days=1"
    )

    try:
        cevap = requests.get(url, timeout=10)
        veri = cevap.json()

        current = veri["current"]
        sicaklik = current["temperature_2m"]
        nem = current["relative_humidity_2m"]
        anlik_yagis = current["rain"]
        saatlik_yagis = veri["hourly"]["rain"]

        if len(saatlik_yagis) >= 3:
            uc_saat_yagis = sum(saatlik_yagis[:3])
        else:
            uc_saat_yagis = anlik_yagis

        return "Güncel veri alındı", sicaklik, nem, anlik_yagis, uc_saat_yagis

    except Exception:
        return "İnternet verisi alınamadı", "-", "-", 0, 0


def renk_sec(seviye):
    if seviye == "Kritik":
        return "#d90429"
    elif seviye == "Yuksek":
        return "#f77f00"
    elif seviye == "Orta":
        return "#fcbf49"
    else:
        return "#2a9d8f"


def risk_deger_sec(seviye):
    if seviye == "Kritik":
        return 100
    elif seviye == "Yuksek":
        return 80
    elif seviye == "Orta":
        return 50
    else:
        return 20


def ai_tahmin_yap(yagis, dere, egim, altyapi, gecmis_sel, nufus):
    yeni_veri = pd.DataFrame([{
        "yagis_miktari": yagis,
        "dere_seviyesi": dere,
        "zemin_egimi": egim,
        "altyapi_durumu": altyapi,
        "gecmis_sel_sayisi": gecmis_sel,
        "nufus_yogunlugu": nufus
    }])

    sinif = sinif_modeli.predict(yeni_veri)[0]
    olasilik = sinif_modeli.predict_proba(yeni_veri)[0]
    guven = max(olasilik) * 100
    risk_puani = regresyon_modeli.predict(yeni_veri)[0]

    return sinif, guven, risk_puani


def ai_aciklama_uret(yagis, dere, altyapi, gecmis_sel):
    nedenler = []

    if yagis >= 80:
        nedenler.append("Yağış miktarı çok yüksek.")
    elif yagis >= 50:
        nedenler.append("Yağış miktarı risk oluşturabilecek seviyede.")

    if dere >= 3:
        nedenler.append("Dere seviyesi kritik seviyeye yakın.")
    elif dere >= 2:
        nedenler.append("Dere seviyesi yükselmiş durumda.")

    if altyapi <= 2:
        nedenler.append("Altyapı durumu yetersiz.")

    if gecmis_sel >= 4:
        nedenler.append("Bölgede geçmiş sel sayısı fazla.")

    if len(nedenler) == 0:
        nedenler.append("Bölgesel veriler şu an ciddi risk göstermiyor.")

    return "\n".join(["• " + n for n in nedenler])


def mudahale_onerisi(seviye):
    if seviye == "Kritik":
        return (
            "Acil tahliye başlatılmalı, AFAD ve belediye ekipleri bölgeye yönlendirilmelidir. "
            "Vatandaşlara uyarı bildirimi gönderilmelidir."
        )
    elif seviye == "Yuksek":
        return (
            "Dere yatakları kontrol edilmeli, su tahliye ekipleri hazır bekletilmelidir. "
            "Riskli sokaklar izlenmelidir."
        )
    elif seviye == "Orta":
        return "Bölge izlenmeli, yağış ve dere seviyesi düzenli takip edilmelidir."
    else:
        return "Normal takip yeterlidir. Şu anda ciddi bir tehlike görünmemektedir."


def secili_bolge_getir():
    bolge = bolge_combo.get()
    if bolge == "":
        messagebox.showwarning("Uyarı", "Lütfen bir bölge seçiniz.")
        return None
    return df[df["bolge"] == bolge].iloc[0]


# =========================================================
# ANALİZ FONKSİYONLARI
# =========================================================
def analiz_et():
    secilen = secili_bolge_getir()
    if secilen is None:
        return

    hava, sicaklik, nem, guncel_yagis, uc_saat_yagis = guncel_hava_cek(
        secilen["enlem"],
        secilen["boylam"]
    )

    ihbar_etkisi = ihbar_seviyesi.get() * 5

    kullanilan_yagis = max(
        float(secilen["yagis_miktari"]),
        float(guncel_yagis)
    ) + ihbar_etkisi

    tahmin, guven, risk_puani = ai_tahmin_yap(
        kullanilan_yagis,
        secilen["dere_seviyesi"],
        secilen["zemin_egimi"],
        secilen["altyapi_durumu"],
        secilen["gecmis_sel_sayisi"],
        secilen["nufus_yogunlugu"]
    )

    uc_saat_sonra_yagis = kullanilan_yagis + uc_saat_yagis

    gelecek_tahmin, gelecek_guven, gelecek_puan = ai_tahmin_yap(
        uc_saat_sonra_yagis,
        secilen["dere_seviyesi"],
        secilen["zemin_egimi"],
        secilen["altyapi_durumu"],
        secilen["gecmis_sel_sayisi"],
        secilen["nufus_yogunlugu"]
    )

    renk = renk_sec(tahmin)

    sonuc_kart.config(bg=renk)
    risk_baslik.config(text=tahmin.upper(), bg=renk)
    risk_puan_label.config(
        text=f"Güven: %{guven:.2f}\nRisk Puanı: {risk_puani:.1f}/100",
        bg=renk
    )

    risk_kart_etiket.config(text=tahmin.upper(), bg=renk)
    risk_kart_puan.config(
        text=f"Güven: %{guven:.2f} | Risk Puanı: {risk_puani:.1f}/100",
        bg=renk
    )

    aciklama = ai_aciklama_uret(
        kullanilan_yagis,
        secilen["dere_seviyesi"],
        secilen["altyapi_durumu"],
        secilen["gecmis_sel_sayisi"]
    )

    oneri = mudahale_onerisi(tahmin)

    detay_text.config(state="normal")
    detay_text.delete("1.0", tk.END)

    metin = f"""
BÖLGE ANALİZİ

Bölge: {secilen['bolge']}

VERİLER
CSV Yağış Miktarı: {secilen['yagis_miktari']} mm
İnternetten Güncel Yağış: {guncel_yagis} mm
Vatandaş İhbar Etkisi: +{ihbar_etkisi} puan
Modelde Kullanılan Yağış: {kullanilan_yagis} mm

Dere Seviyesi: {secilen['dere_seviyesi']} m
Zemin Eğimi: {secilen['zemin_egimi']}
Altyapı Durumu: {secilen['altyapi_durumu']}
Geçmiş Sel Sayısı: {secilen['gecmis_sel_sayisi']}
Nüfus Yoğunluğu: {secilen['nufus_yogunlugu']}

GÜNCEL HAVA
Durum: {hava}
Sıcaklık: {sicaklik} °C
Nem: %{nem}

YAPAY ZEKA MODELİ
Sınıflandırma Modeli: Random Forest Classifier
Regresyon Modeli: Random Forest Regressor
Doğruluk: %{dogruluk * 100:.2f}
F1-Score: %{f1 * 100:.2f}
Regresyon Ortalama Hata: {mae:.2f}

ANLIK SONUÇ
Tahmin Edilen Risk Seviyesi: {tahmin}
Model Güven Oranı: %{guven:.2f}
Risk Puanı: {risk_puani:.1f}/100

3 SAAT SONRA TAHMİN
Tahmini Risk Seviyesi: {gelecek_tahmin}
Tahmini Güven: %{gelecek_guven:.2f}
Tahmini Risk Puanı: {gelecek_puan:.1f}/100

AI AÇIKLAMA
{aciklama}

MÜDAHALE ÖNERİSİ
{oneri}
"""

    detay_text.insert(tk.END, metin)
    detay_text.config(state="disabled")

    if tahmin == "Kritik":
        messagebox.showwarning(
            "Acil Uyarı",
            "Kritik sel riski tespit edildi.\nVatandaşlara acil uyarı gönderildi."
        )


def risk_grafigi_goster():
    risk_sayilari = df["model_tahmini"].value_counts()

    plt.figure(figsize=(8, 5))
    plt.bar(risk_sayilari.index, risk_sayilari.values)
    plt.title("Random Forest Modeline Göre Risk Sınıfları")
    plt.xlabel("Risk Seviyesi")
    plt.ylabel("Bölge Sayısı")
    plt.tight_layout()
    plt.show()


def ozellik_onemi_goster():
    onemler = sinif_modeli.feature_importances_

    plt.figure(figsize=(9, 5))
    plt.bar(ozellikler, onemler)
    plt.title("Explainable AI - Random Forest Özellik Önemi")
    plt.xlabel("Özellikler")
    plt.ylabel("Önem Değeri")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def detay_grafigi_goster():
    secilen = secili_bolge_getir()
    if secilen is None:
        return

    bolge = secilen["bolge"]

    etiketler = ["Yağış", "Dere", "Eğim", "Altyapı", "Geçmiş Sel", "Nüfus"]
    degerler = [
        secilen["yagis_miktari"],
        secilen["dere_seviyesi"] * 20,
        secilen["zemin_egimi"] * 10,
        secilen["altyapi_durumu"] * 10,
        secilen["gecmis_sel_sayisi"] * 10,
        secilen["nufus_yogunlugu"] * 10
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(etiketler, degerler)
    plt.title(f"{bolge} Bölgesi Risk Etkenleri")
    plt.ylabel("Etki Değeri")
    plt.tight_layout()
    plt.show()


# =========================================================
# HARİTA - SENİN ATTIĞIN HTML HARİTAYI AÇAR
# =========================================================
def harita_goster():
    """
    Eski Folium haritası kaldırıldı.
    Bu fonksiyon, aynı klasörde bulunan özel HTML haritasını açar.
    """
    harita_dosyasi = "samsun_sel_risk_haritasi-5.html"
    dosya_yolu = os.path.abspath(harita_dosyasi)

    if not os.path.exists(dosya_yolu):
        messagebox.showerror(
            "Harita Bulunamadı",
            f"{harita_dosyasi} bulunamadı.\n\n"
            "Lütfen bu HTML dosyasını main.py ile aynı klasöre koy."
        )
        return

    webbrowser.open("file://" + dosya_yolu)


# =========================================================
# OPERASYON TABLOLARI
# =========================================================
def kritik_bolgeler_goster():
    kritikler = df[df["model_tahmini"] == "Kritik"]

    yeni = tk.Toplevel(root)
    yeni.title("Öncelikli Müdahale Listesi")
    yeni.geometry("900x430")
    yeni.configure(bg="#edf6f9")

    tablo = ttk.Treeview(
        yeni,
        columns=("sira", "bolge", "tahmin", "puan", "yagis", "dere"),
        show="headings"
    )

    tablo.heading("sira", text="Öncelik")
    tablo.heading("bolge", text="Bölge")
    tablo.heading("tahmin", text="AI Tahmini")
    tablo.heading("puan", text="Risk Puanı")
    tablo.heading("yagis", text="Yağış")
    tablo.heading("dere", text="Dere Seviyesi")

    tablo.pack(fill="both", expand=True, padx=15, pady=15)

    sirali = kritikler.sort_values(
        by=["model_risk_puani", "yagis_miktari", "dere_seviyesi"],
        ascending=False
    )

    for i in range(len(sirali)):
        tablo.insert(
            "",
            tk.END,
            values=(
                i + 1,
                sirali.iloc[i]["bolge"],
                sirali.iloc[i]["model_tahmini"],
                sirali.iloc[i]["model_risk_puani"],
                sirali.iloc[i]["yagis_miktari"],
                sirali.iloc[i]["dere_seviyesi"]
            )
        )


def tablo_goster():
    yeni = tk.Toplevel(root)
    yeni.title("AI Risk Tablosu")
    yeni.geometry("1000x450")
    yeni.configure(bg="#edf6f9")

    tablo = ttk.Treeview(
        yeni,
        columns=("bolge", "gercek", "tahmin", "puan", "yagis", "dere"),
        show="headings"
    )

    tablo.heading("bolge", text="Bölge")
    tablo.heading("gercek", text="CSV Risk")
    tablo.heading("tahmin", text="AI Tahmini")
    tablo.heading("puan", text="Risk Puanı")
    tablo.heading("yagis", text="Yağış")
    tablo.heading("dere", text="Dere")

    tablo.pack(fill="both", expand=True, padx=15, pady=15)

    for i in range(len(df)):
        tablo.insert(
            "",
            tk.END,
            values=(
                df.iloc[i]["bolge"],
                df.iloc[i]["risk_sinifi"],
                df.iloc[i]["model_tahmini"],
                df.iloc[i]["model_risk_puani"],
                df.iloc[i]["yagis_miktari"],
                df.iloc[i]["dere_seviyesi"]
            )
        )


# =========================================================
# GELİŞTİRME 7: SEKMELİ ARAYÜZ
# =========================================================
root = tk.Tk()
root.title("FloodGuard AI - Afet Yönetim Merkezi")
root.geometry("1160x780")
root.configure(bg="#edf6f9")

style = ttk.Style()
style.theme_use("clam")
style.configure(
    "TNotebook",
    background="#edf6f9",
    borderwidth=0
)
style.configure(
    "TNotebook.Tab",
    font=("Arial", 12, "bold"),
    padding=[18, 10]
)
style.map(
    "TNotebook.Tab",
    background=[("selected", "#023047")],
    foreground=[("selected", "white")]
)
style.configure(
    "Treeview.Heading",
    font=("Arial", 11, "bold")
)

header = tk.Frame(root, bg="#023047", height=105)
header.pack(fill="x")

baslik = tk.Label(
    header,
    text="🌊 FloodGuard AI",
    font=("Arial", 32, "bold"),
    bg="#023047",
    fg="white"
)
baslik.pack(pady=(16, 0))

alt_baslik = tk.Label(
    header,
    text="Sekmeli Afet Yönetim Merkezi | AI Sel Erken Uyarı ve Karar Destek Sistemi",
    font=("Arial", 14),
    bg="#023047",
    fg="#caf0f8"
)
alt_baslik.pack()

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, padx=16, pady=14)

ana_tab = tk.Frame(notebook, bg="#edf6f9")
harita_tab = tk.Frame(notebook, bg="#edf6f9")
grafik_tab = tk.Frame(notebook, bg="#edf6f9")
model_tab = tk.Frame(notebook, bg="#edf6f9")
operasyon_tab = tk.Frame(notebook, bg="#edf6f9")

notebook.add(ana_tab, text="🏠 Ana Analiz")
notebook.add(harita_tab, text="🗺 Harita")
notebook.add(grafik_tab, text="📊 Grafikler")
notebook.add(model_tab, text="🤖 Model")
notebook.add(operasyon_tab, text="🚨 Operasyon")

# ANA ANALİZ TAB
sol_panel = tk.Frame(ana_tab, bg="white", bd=0)
sol_panel.place(x=20, y=20, width=455, height=570)

orta_panel = tk.Frame(ana_tab, bg="#edf6f9")
orta_panel.place(x=500, y=20, width=250, height=250)

sag_panel = tk.Frame(ana_tab, bg="white")
sag_panel.place(x=775, y=20, width=330, height=570)

tk.Label(
    sol_panel,
    text="📍 Bölge Analizi",
    font=("Arial", 20, "bold"),
    bg="white",
    fg="#023047"
).pack(pady=(18, 8))

bolge_combo = ttk.Combobox(
    sol_panel,
    values=list(df["bolge"]),
    font=("Arial", 13),
    width=27,
    state="readonly"
)
bolge_combo.pack(pady=8)

tk.Label(
    sol_panel,
    text="Vatandaş İhbar Seviyesi",
    font=("Arial", 13, "bold"),
    bg="white"
).pack(pady=(8, 2))

ihbar_seviyesi = tk.Scale(
    sol_panel,
    from_=0,
    to=5,
    orient="horizontal",
    bg="white",
    length=280
)
ihbar_seviyesi.pack()

tk.Button(
    sol_panel,
    text="AI ile Analiz Et",
    font=("Arial", 13, "bold"),
    bg="#0077b6",
    fg="white",
    width=24,
    height=2,
    command=analiz_et
).pack(pady=10)

detay_text = tk.Text(
    sol_panel,
    width=47,
    height=22,
    font=("Arial", 11),
    bg="#f8f9fa",
    fg="#212529",
    bd=0,
    state="disabled"
)
detay_text.pack(padx=10, pady=8)

sonuc_kart = tk.Frame(
    orta_panel,
    bg="#adb5bd",
    width=245,
    height=230
)
sonuc_kart.pack()
sonuc_kart.pack_propagate(False)

tk.Label(
    sonuc_kart,
    text="AI RİSK TAHMİNİ",
    font=("Arial", 14, "bold"),
    bg="#adb5bd",
    fg="white"
).pack(pady=(25, 5))

risk_baslik = tk.Label(
    sonuc_kart,
    text="BÖLGE\nSEÇİN",
    font=("Arial", 27, "bold"),
    bg="#adb5bd",
    fg="white"
)
risk_baslik.pack()

risk_puan_label = tk.Label(
    sonuc_kart,
    text="Güven: -\nRisk Puanı: -",
    font=("Arial", 14, "bold"),
    bg="#adb5bd",
    fg="white"
)
risk_puan_label.pack(pady=5)

tk.Label(
    sag_panel,
    text="🧭 Hızlı İşlemler",
    font=("Arial", 20, "bold"),
    bg="white",
    fg="#023047"
).pack(pady=(20, 15))

hizli_butonlar = [
    ("🗺 Gelişmiş Risk Haritası", harita_goster, "#8ecae6"),
    ("📊 Risk Grafiği", risk_grafigi_goster, "#ffb703"),
    ("📈 Bölge Detay Grafiği", detay_grafigi_goster, "#90be6d"),
    ("🤖 Özellik Önemi", ozellik_onemi_goster, "#bde0fe"),
    ("🚨 Öncelikli Müdahale", kritik_bolgeler_goster, "#ffc8dd"),
    ("📋 Risk Tablosu", tablo_goster, "#adb5bd")
]

for text, komut, renk in hizli_butonlar:
    tk.Button(
        sag_panel,
        text=text,
        font=("Arial", 12, "bold"),
        bg=renk,
        fg="black",
        width=27,
        height=2,
        command=komut
    ).pack(pady=6)

# HARİTA TAB
harita_kutu = tk.Frame(harita_tab, bg="white")
harita_kutu.pack(fill="both", expand=True, padx=30, pady=30)

tk.Label(
    harita_kutu,
    text="🗺 Gelişmiş AI Sel Risk Haritası",
    font=("Arial", 25, "bold"),
    bg="white",
    fg="#023047"
).pack(pady=(35, 10))

tk.Label(
    harita_kutu,
    text=(
        "Bu haritada ısı haritası, uydu görünümü, risk renkleri, kritik bölge uyarı işaretleri, "
        "mini harita ve müdahale önerili popup ekranları bulunur."
    ),
    font=("Arial", 14),
    bg="white",
    fg="#495057",
    wraplength=850,
    justify="center"
).pack(pady=10)

tk.Button(
    harita_kutu,
    text="Haritayı Tarayıcıda Aç",
    font=("Arial", 15, "bold"),
    bg="#0077b6",
    fg="white",
    width=28,
    height=2,
    command=harita_goster
).pack(pady=25)

# GRAFİK TAB
grafik_kutu = tk.Frame(grafik_tab, bg="white")
grafik_kutu.pack(fill="both", expand=True, padx=30, pady=30)

tk.Label(
    grafik_kutu,
    text="📊 Grafik Merkezi",
    font=("Arial", 25, "bold"),
    bg="white",
    fg="#023047"
).pack(pady=(35, 15))

grafik_buton_frame = tk.Frame(grafik_kutu, bg="white")
grafik_buton_frame.pack(pady=15)

grafik_butonlari = [
    ("AI Risk Sınıfları Grafiği", risk_grafigi_goster, "#ffb703"),
    ("Seçili Bölge Detay Grafiği", detay_grafigi_goster, "#90be6d"),
    ("Model Özellik Önemi Grafiği", ozellik_onemi_goster, "#bde0fe")
]

for text, komut, renk in grafik_butonlari:
    tk.Button(
        grafik_buton_frame,
        text=text,
        font=("Arial", 13, "bold"),
        bg=renk,
        fg="black",
        width=32,
        height=2,
        command=komut
    ).pack(pady=9)

# MODEL TAB
model_kutu = tk.Frame(model_tab, bg="white")
model_kutu.pack(fill="both", expand=True, padx=30, pady=30)

tk.Label(
    model_kutu,
    text="🤖 Model Performans Paneli",
    font=("Arial", 25, "bold"),
    bg="white",
    fg="#023047"
).pack(pady=(30, 20))

risk_kart = tk.Frame(model_kutu, bg="#adb5bd", width=550, height=120)
risk_kart.pack(pady=10)
risk_kart.pack_propagate(False)

risk_kart_etiket = tk.Label(
    risk_kart,
    text="ANALİZ BEKLENİYOR",
    font=("Arial", 24, "bold"),
    bg="#adb5bd",
    fg="white"
)
risk_kart_etiket.pack(pady=(25, 5))

risk_kart_puan = tk.Label(
    risk_kart,
    text="Güven: - | Risk Puanı: -",
    font=("Arial", 13, "bold"),
    bg="#adb5bd",
    fg="white"
)
risk_kart_puan.pack()

performans_metin = f"""
Kullanılan Sınıflandırma Modeli: Random Forest Classifier
Kullanılan Regresyon Modeli: Random Forest Regressor

Doğruluk Oranı: %{dogruluk * 100:.2f}
F1-Score: %{f1 * 100:.2f}
Regresyon Ortalama Hata MAE: {mae:.2f}

Modelin Kullandığı Özellikler:
• Yağış miktarı
• Dere seviyesi
• Zemin eğimi
• Altyapı durumu
• Geçmiş sel sayısı
• Nüfus yoğunluğu
"""

tk.Label(
    model_kutu,
    text=performans_metin,
    font=("Arial", 14),
    bg="white",
    fg="#212529",
    justify="left"
).pack(pady=15)

tk.Button(
    model_kutu,
    text="Model Özellik Önemini Göster",
    font=("Arial", 13, "bold"),
    bg="#bde0fe",
    fg="black",
    width=30,
    height=2,
    command=ozellik_onemi_goster
).pack(pady=10)

# OPERASYON TAB
operasyon_kutu = tk.Frame(operasyon_tab, bg="white")
operasyon_kutu.pack(fill="both", expand=True, padx=30, pady=30)

tk.Label(
    operasyon_kutu,
    text="🚨 Operasyon ve Müdahale Paneli",
    font=("Arial", 25, "bold"),
    bg="white",
    fg="#023047"
).pack(pady=(35, 15))

tk.Label(
    operasyon_kutu,
    text="Kritik bölgeleri öncelik sırasına göre görüntüleyebilir veya tüm AI risk tablosunu açabilirsin.",
    font=("Arial", 14),
    bg="white",
    fg="#495057",
    wraplength=850,
    justify="center"
).pack(pady=10)

tk.Button(
    operasyon_kutu,
    text="Öncelikli Müdahale Listesini Aç",
    font=("Arial", 14, "bold"),
    bg="#ffc8dd",
    fg="black",
    width=34,
    height=2,
    command=kritik_bolgeler_goster
).pack(pady=12)

tk.Button(
    operasyon_kutu,
    text="AI Risk Tablosunu Aç",
    font=("Arial", 14, "bold"),
    bg="#adb5bd",
    fg="black",
    width=34,
    height=2,
    command=tablo_goster
).pack(pady=12)

# FOOTER
footer = tk.Label(
    root,
    text=(
        f"FloodGuard AI | CSV: {csv_yolu} | Random Forest Classifier + Regressor | "
        f"Open-Meteo | Doğruluk: %{dogruluk * 100:.2f} | F1-Score: %{f1 * 100:.2f} | MAE: {mae:.2f}"
    ),
    font=("Arial", 10),
    bg="#edf6f9",
    fg="#6c757d"
)
footer.pack(side="bottom", pady=6)

root.mainloop()
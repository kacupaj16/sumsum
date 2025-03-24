import os
import cv2
import requests
import moviepy as mp
from srt import parse
from bs4 import BeautifulSoup
from pydub import AudioSegment
from PIL import Image, ImageDraw, ImageFont


def get_text_from_srt(filename:str ):
    
    srtfile = open(filename, "r", encoding="utf-8")
    text = srtfile.read()
    srtfile.close()
    subs = parse(text)
    all_subs = ''
    for s in subs:
        #start = s.start.seconds + s.start.microseconds/1000000
        #end  = s.end.seconds + s.end.microseconds/1000000
        all_subs+=s.content.replace('\n',' ').replace('<i>','').replace('</i>','')
        all_subs+='\n'
    return all_subs

def get_subs_from_srt(filename:str ):
    srtfile = open(filename, "r")
    text = srtfile.read()
    srtfile.close()
    subs = parse(text)
    subs_lst= []
    for s in subs:
        subs_lst.append(s)
    return subs_lst


def write_subs_to_video(video_fn:str,subs_fn:str,output_path:str='sample_with_subs_exp.mp4',font_sz:int=40,max_length:int=70):
    #if len(line)>upper_limit: insert /n at prev space
    subs = get_subs_from_srt(subs_fn)
    video = mp.VideoFileClip(video_fn)

    fps = int(round(video.fps))
    print(fps)
    audio = AudioSegment.from_file(video_fn,video_fn[-3:])
    audio.export('temp.mp3')

    frame_count = 0
    subs_index=0
    clips = []
    for frame in video.iter_frames():
        #print(frame_count)
        #plt.imshow(frame)
        #plt.show()
        try:
            s = subs[subs_index]

            start_s = int(round((s.start.seconds + (s.start.microseconds)/1000000)* fps)) 
            end_s = int(round((s.end.seconds + (s.end.microseconds)/1000000) * fps ))

            if end_s>frame_count and start_s<=frame_count:
                subtitle_text = s.content
            elif end_s==frame_count:
                subtitle_text = s.content
                subs_index+=1
            else:
                subtitle_text = ''

            if len(subtitle_text)>0:
                
                subtitle_text = break_chunk(subtitle_text,max_length=max_length)
                pil_image = Image.fromarray(frame)
                font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", font_sz, encoding="unic")
                draw = ImageDraw.Draw(pil_image)
                draw.text((30, 30), f'{subtitle_text}', font=font,fill="#FF00FF")
                frame = np.asarray(pil_image)
            clips.append(frame)
        except IndexError:
            pass
            #print(f'Out of subs.Lasts timestamp:{end_s}')

        frame_count+=1

    size = video.size
    temp_file = 'temp.mp4'
    out = cv2.VideoWriter(temp_file,cv2.VideoWriter_fourcc('m','p','4','v'), fps, size)
    for f in clips:
        out.write( cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    out.release()
    video_noa = mp.VideoFileClip(temp_file)
    audio_noa = mp.AudioFileClip('temp.mp3')


    video_noa = video_noa.set_audio(audio_noa)
    print(f'Saving file at :\t{output_path}')
    video_noa.write_videofile(output_path)

    os.remove('temp.mp3')
    os.remove('temp.mp4')

def break_chunk(chunk:str,max_length:int=50):
    positions = []
    mid = max_length//2
    min_diff = 1000000

    if len(chunk)>max_length:
        for i in range(len(chunk)):
            if chunk[i]=='_':
                positions.append(i)
        if len(positions)==0: #no space char found
            new_chunk = chunk[:mid] + '\n-' + chunk[mid:]
            return new_chunk
        else:
            for p in positions:
                if abs(mid-p)<min_diff:
                    min_diff=p
            new_chunk = chunk[:p] + '\n' + chunk[p:]
    else: 
        return chunk
    
def get_title_video(url_video:str='https://youtu.be/DyRjpoBL9aI?si=5msxGmyZ2k64hKzz'):
    r = requests.get(url_video)
    soup = BeautifulSoup(r.text)

    link = soup.find_all(name="title")[0]
    title = str(link)
    title = title.replace("<title>","")
    title = title.replace("</title>","")

    return title



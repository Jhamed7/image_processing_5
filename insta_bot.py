import glob
from instabot import Bot  # noqa: E402

print('This app will post a picture to instagram for you\n')
caption = input('Please insert your caption : ')
# username = input('\nInsert your instagram username : ')
# password = input('\nInsert your password : ')


bot = Bot()
bot.login()

while True:
    folder_path = "./insta_pics"
    pics = glob.glob(folder_path + "/*.jpg")
    pics = sorted(pics)
    try:
        for pic in pics:

            pic_name = pic[:-4].split("-")
            pic_name = "-".join(pic_name[1:])

            bot.upload_photo(pic, caption=caption)
            if bot.api.last_response.status_code != 200:
                print(bot.api.last_response)
                # snd msg
                break

    except Exception as e:
        print(str(e))

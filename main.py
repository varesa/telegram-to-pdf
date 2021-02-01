import os
import logging
import tempfile
import subprocess
import requests
import telegram
from telegram.ext import CommandHandler, MessageHandler, Updater, Filters
from image import fix_image

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)


def start(update, context):
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Hello. Please start by sending one or more pictures, followed by /roll - "
             "Source: https://github.com/varesa/telegram-to-pdf (BETA)")


def get_largest_photo(photo_sizes):
    max_w = None
    max_photo = None
    for size in photo_sizes:
        if not max_w or size.width > max_w:
            max_w = size.width
            max_photo = size
    return max_photo


chat_pictures = {}


def message(update, context):
    chat_id = update.effective_chat.id
    if chat_id not in chat_pictures.keys():
        chat_pictures[chat_id] = []

    file = None
    if document := update.message.document:
        file = document.get_file()
    if photo_sizes := update.message.photo:
        file = get_largest_photo(photo_sizes).get_file()
    if file:
        chat_pictures[chat_id].append(file.file_path)

    print(update)


def download_file(url, path):
    response = requests.get(url)
    with open(path, 'wb') as file:
        file.write(response.content)
    return path


def make_pdf(original_file):
    pdf_file = original_file + '.pdf'
    subprocess.call(['convert', original_file, '-density', '300', pdf_file])
    return pdf_file


def process_job(urls, bot, chat_id):
    with tempfile.TemporaryDirectory() as temp:
        bot.send_message(chat_id=chat_id, text='Processing:')
        pdf_pages = []
        for index, url in enumerate(urls):
            original_image = download_file(url, os.path.join(temp, str(index)))
            fixed_image = original_image + '-fix.png'
            fix_image(original_image, fixed_image)
            with open(fixed_image, 'rb') as f:
                bot.send_photo(chat_id=chat_id, photo=f)
            pdf_pages.append(make_pdf(fixed_image))
        pdf_file = os.path.join(temp, 'out.pdf')
        subprocess.call(['qpdf', '--empty', '--pages'] + pdf_pages + ['--', pdf_file])
        with open(pdf_file, 'rb') as f:
            bot.send_document(chat_id=chat_id, document=f)


def roll(update, context):
    chat_id = update.effective_chat.id
    if chat_id in chat_pictures.keys():
        urls = chat_pictures[chat_id]
        process_job(urls, context.bot, chat_id)
        chat_pictures[chat_id].clear()


def main():
    updater = Updater(token=os.environ['TOKEN'])
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler('start', start))
    dispatcher.add_handler(CommandHandler('roll', roll))
    dispatcher.add_handler(MessageHandler(Filters.all, message))
    updater.start_polling()


if __name__ == '__main__':
    main()

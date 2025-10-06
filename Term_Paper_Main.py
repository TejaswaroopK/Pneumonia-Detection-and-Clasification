import Term_Paper_VGG as cm
import Term_Paper_ImgGen as ic
from matplotlib import pyplot as plt
if __name__ == "__main__":
    images_folder_path = 'train'
    imgdg = ic.ImgDG()
    imgdg.visualize(images_folder_path, nimages=2)
    image_df, train, label = imgdg.preprocess(images_folder_path)
    image_df.to_csv("image_df.csv")
    tr_gen, tt_gen, va_gen = imgdg.generate_train_test_images(image_df, train, label)
    print("Length of Test Data Generated : ",len(tt_gen))

    # CNN model
    # Create an instance of the custom model
    cnn_model = cm.CNN.cnn_vgg()
    # CNN model
    # Create an instance of the custom model

    # Compile the model
    cnn_model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    # Train the model
    history = cnn_model.fit(tr_gen, epochs=5, validation_data=va_gen)

    # Evaluate the model
    Cnn_test_loss, Cnn_test_acc = cnn_model.evaluate(tt_gen)
    print(f'Test accuracy: {Cnn_test_acc}')
    print(cnn_model.summary())

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

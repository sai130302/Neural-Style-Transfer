import utils.utils as utils
import neural_style_transfer as NST
content_img = utils.get_image("dancing.jpg",(512,512))
style_img = utils.get_image("picasso.jpg",(512,512))
input_img = content_img.clone()

input_img = NST.run_style_transfer(content_img,style_img,input_img)
utils.tensor_to_image(input_img,True,"Styled Image")
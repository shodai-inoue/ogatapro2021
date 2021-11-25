var y0 = 0;
var y = 0;
$(function(){
  $(window).on('load scroll',function() {
    if ($('.hamburger').hasClass('active')) {
      ;//何もしない
    } else {
      y = $(this).scrollTop();
      if (y > 540 && y >= y0) {
        $('.sitemenu').addClass('hide');
      } else {
        $('.sitemenu').removeClass('hide');
      }
      y0 = y;
    }
  });
});

const CLASSNAME = "-visible";
const TIMEOUT = 1500;
const $target = $(".title");

setInterval(() => {
  $target.addClass(CLASSNAME);
  setTimeout(() => {
    $target.removeClass(CLASSNAME);
  }, TIMEOUT);
}, TIMEOUT * 2);

// $(function() {
// 	setTimeout(function(){
// 		$('.start p').fadeIn(1600);
// 	},500); //0.5秒後にロゴをフェードイン!
// 	setTimeout(function(){
// 		$('.start').fadeOut(500);
// 	},2500); //2.5秒後にロゴ含め真っ白背景をフェードアウト！
// });

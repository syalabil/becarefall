package com.example.be_care_fall;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.NotificationCompat;
import androidx.core.app.NotificationManagerCompat;

import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.StrictMode;
import android.telephony.SmsManager;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.MediaController;
import android.widget.Toast;
import android.widget.VideoView;

import java.io.IOException;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.FormBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class MainActivity extends AppCompatActivity {

    Camera camera;
    VideoView video;
    ShowCamera showCamera;

    Handler handler = new Handler();
    Runnable runnable;
    int delay = 1000;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.O){
            NotificationChannel channel = new NotificationChannel("My Notification","My Notification", NotificationManager.IMPORTANCE_DEFAULT);
            NotificationManager manager = getSystemService(NotificationManager.class);
            manager.createNotificationChannel(channel);
        }

//        //Onclick for button to launch application
//        Button btn =findViewById(R.id.runapplcation);
//        btn.setOnClickListener(new View.OnClickListener(){
//            @Override
//            public void onClick(View view){
//                setContentView(R.layout.application);
//            }
//        });

        //Set Up Camera
//        camera = Camera.open();
//        showCamera = new ShowCamera(this, camera);
//        frameLayout.addView(showCamera);


        //Onclick for button to call ambulance
        Button btn = findViewById(R.id.ambulance);
        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Intent.ACTION_DIAL);
                intent.setData(Uri.parse("tel:87385624"));
                startActivity(intent);
            }
        });
    }
    @Override
    protected void onResume() {
        handler.postDelayed(runnable = new Runnable() {
            public void run() {
                handler.postDelayed(runnable, delay);
                OkHttpClient okHttpClient = new OkHttpClient();
                OkHttpClient okHttpClient2 = new OkHttpClient();
                OkHttpClient okHttpClient3 = new OkHttpClient();
//              RequestBody formbody = new FormBody.Builder().add("value","please repeat this!").build();
                Request request = new Request.Builder().url("https://5ed2-115-66-189-103.ngrok.io/result").build();

                Request request2 = new Request.Builder().url("https://5ed2-115-66-189-103.ngrok.io/sms_result").build();

                Request request3 = new Request.Builder().url("https://5ed2-115-66-189-103.ngrok.io/sendurl").build();

                // Is for Notification
                okHttpClient.newCall(request).enqueue(new Callback() {
                    @Override
                    public void onFailure(@NonNull Call call, @NonNull IOException e) {
                        NotificationCompat.Builder builder = new NotificationCompat.Builder(MainActivity.this,"My Notification");
                        builder.setSmallIcon(R.drawable.photo_2022_04_26_11_54_04);
                        builder.setContentTitle("ALERT !!!");
                        builder.setContentText("Fail To Connect for notification");
                        builder.setAutoCancel(true);

                        NotificationManagerCompat managerCompat = NotificationManagerCompat.from(MainActivity.this);
                        managerCompat.notify(1,builder.build());





                    }

                    @Override
                    public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                        if(response.body().string().equals("fall")) {
                            NotificationCompat.Builder builder = new NotificationCompat.Builder(MainActivity.this, "My Notification");
                            builder.setSmallIcon(R.drawable.photo_2022_04_26_11_54_04);
                            builder.setContentTitle("OH NO !!!");
                            builder.setContentText("Someone fell");
                            builder.setAutoCancel(true);

                            NotificationManagerCompat managerCompat = NotificationManagerCompat.from(MainActivity.this);
                            managerCompat.notify(1, builder.build());

                            // Is for Video View
                            okHttpClient3.newCall(request3).enqueue(new Callback() {
                                @Override
                                public void onFailure(@NonNull Call call, @NonNull IOException e) {
                                    NotificationCompat.Builder builder = new NotificationCompat.Builder(MainActivity.this,"My Notification");
                                    builder.setSmallIcon(R.drawable.photo_2022_04_26_11_54_04);
                                    builder.setContentTitle("ALERT !!!");
                                    builder.setContentText("Fail To Connect for Video");
                                    builder.setAutoCancel(true);

                                    NotificationManagerCompat managerCompat = NotificationManagerCompat.from(MainActivity.this);
                                    managerCompat.notify(1,builder.build());

                                }

                                @Override
                                public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
//                                    video = findViewById(R.id.video);
////                                  String videoUrl = response.body().string();
//                                    String videoUrl = "https://storage.googleapis.com/android_python_api_bucket/video2.mp4";
//                                    // resource from the videoUrl
//                                    Uri uri = Uri.parse(videoUrl);
//
//                                    // sets the resource from the
//                                    // videoUrl to the videoView
//                                    video.setVideoURI(uri);
//
//                                    MediaController mediaController = new MediaController(MainActivity.this);
//
//                                    // sets the anchor view
//                                    // anchor view for the videoView
//                                    mediaController.setAnchorView(video);
//
//                                    // sets the media player to the videoView
//                                    mediaController.setMediaPlayer(video);
//
//                                    // sets the media controller to the videoView
//                                    video.setMediaController(mediaController);
//
//                                    // starts the video
//                                    video.start();
//

                                }
                            });
                        }
                    }
                });

                // Is for SMS
                okHttpClient2.newCall(request2).enqueue(new Callback() {
                    @Override
                    public void onFailure(@NonNull Call call, @NonNull IOException e) {
                        NotificationCompat.Builder builder = new NotificationCompat.Builder(MainActivity.this,"My Notification");
                        builder.setSmallIcon(R.drawable.photo_2022_04_26_11_54_04);
                        builder.setContentTitle("ALERT !!!");
                        builder.setContentText("Fail To Connect for sms");
                        builder.setAutoCancel(true);

                        NotificationManagerCompat managerCompat = NotificationManagerCompat.from(MainActivity.this);
                        managerCompat.notify(1,builder.build());

                    }

                    @Override
                    public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                        if(response.body().string().equals("confirm")) {
                            SmsManager smsManager = SmsManager.getDefault();
                            smsManager.sendTextMessage("97212263",null,"We need help, someone is dying",null,null);


                            NotificationCompat.Builder builder = new NotificationCompat.Builder(MainActivity.this, "My Notification");
                            builder.setSmallIcon(R.drawable.photo_2022_04_26_11_54_04);
                            builder.setContentTitle("Alert !!!");
                            builder.setContentText("We have called the ambulance regarding a fall");
                            builder.setAutoCancel(true);

                            NotificationManagerCompat managerCompat = NotificationManagerCompat.from(MainActivity.this);
                            managerCompat.notify(1, builder.build());

                        }
                    }
                });


            }
        }, delay);
        super.onResume();
    }
    @Override
    protected void onPause() {
        super.onPause();
        handler.removeCallbacks(runnable); //stop handler when activity not visible super.onPause();
    }
}
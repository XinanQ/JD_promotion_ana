import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
## 量化不同促销方式对订单金额的影响，找出哪种促销方式最能提升销售表现
# 特征工程
def create_features(df):
    # 基础特征
    features = pd.DataFrame()
    
    # 用户特征
    features['user_purchase_count'] = df.groupby('user_ID')['order_ID'].transform('count')
    features['user_total_amount'] = df.groupby('user_ID')['final_unit_price'].transform('sum')
    features['user_avg_amount'] = features['user_total_amount'] / features['user_purchase_count']
    
    # 商品特征
    features['sku_purchase_count'] = df.groupby('sku_ID')['order_ID'].transform('count')
    features['sku_avg_price'] = df.groupby('sku_ID')['final_unit_price'].transform('mean')
    
    # 促销特征
    features['has_direct_discount'] = (df['direct_discount_per_unit'] > 0).astype(int)
    features['has_quantity_discount'] = (df['quantity_discount_per_unit'] > 0).astype(int)
    features['has_bundle_discount'] = (df['bundle_discount_per_unit'] > 0).astype(int)
    features['has_coupon_discount'] = (df['coupon_discount_per_unit'] > 0).astype(int)
    features['has_gift'] = (df['gift_item'].notna()).astype(int)
    
    # 折扣金额特征
    features['total_discount'] = (df['direct_discount_per_unit'] + 
                                df['quantity_discount_per_unit'] + 
                                df['bundle_discount_per_unit'] + 
                                df['coupon_discount_per_unit'])
    features['discount_rate'] = features['total_discount'] / df['original_unit_price']
    
    # 时间特征
    df['order_datetime'] = pd.to_datetime(df['order_time'], errors='coerce')
    features['hour'] = df['order_datetime'].dt.hour
    features['day_of_week'] = df['order_datetime'].dt.dayofweek
    features['is_weekend'] = features['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 价格特征
    features['original_price'] = df['original_unit_price']
    features['final_price'] = df['final_unit_price']
    features['price_per_unit'] = df['final_unit_price'] / df['quantity']
    
    # 促销效果特征
    features['promotion_effect'] = df['quantity'] * features['total_discount']  # 促销带来的总优惠金额
    
    return features

# 分析不同促销方式的效果
def analyze_promotion_effectiveness(df):
    # 创建促销类型组合
    def get_promotion_type(row):
        types = []
        if row['direct_discount_per_unit'] > 0:
            types.append('direct_discount')
        if row['quantity_discount_per_unit'] > 0:
            types.append('quantity_discount')
        if row['bundle_discount_per_unit'] > 0:
            types.append('bundle_discount')
        if row['coupon_discount_per_unit'] > 0:
            types.append('coupon_discount')
        if row['gift_item'] != 0:
            types.append('gift')
        return '+'.join(types) if types else 'no_promotion'

    df['promotion_type'] = df.apply(get_promotion_type, axis=1)
    
    # 按促销类型分组统计
    promotion_stats = df.groupby('promotion_type').agg({
        'final_unit_price': ['mean', 'sum', 'count'],
        'original_unit_price': 'mean',
        'quantity': 'mean',
        'total_discount': 'mean'
    }).round(2)
    
    # 计算每种促销方式的转化率
    promotion_stats['conversion_rate'] = (promotion_stats[('final_unit_price', 'count')] / 
                                        df.groupby('promotion_type')['user_ID'].nunique()).round(4)
    
    # 计算每种促销方式的客单价
    promotion_stats['average_order_value'] = (promotion_stats[('final_unit_price', 'sum')] / 
                                            promotion_stats[('final_unit_price', 'count')]).round(2)
    
    return promotion_stats

# 主函数
def main():
    # 读取数据
    df = pd.read_csv('JD_order_data.csv')
    
    # 特征工程
    features = create_features(df)
    df['total_discount'] = features['total_discount']
    
    # 分析促销效果
    promotion_stats = analyze_promotion_effectiveness(df)
    print("\nPromotion Effectiveness Analysis:")
    print(promotion_stats)
    
    # 输出为CSV文件，便于查看
    promotion_stats.to_csv('promotion_effectiveness_analysis.csv')
    
    # 取订单数最多的Top 5促销类型
    order_counts = promotion_stats[('final_unit_price', 'count')]
    top5_types = order_counts.sort_values(ascending=False).head(5).index
    promotion_stats_top5 = promotion_stats.loc[top5_types]

    plt.figure(figsize=(14, 8))

    # 1. Average order amount by promotion type (Top 5)
    plt.subplot(2, 2, 1)
    promotion_stats_top5[('final_unit_price', 'mean')].plot(kind='bar')
    plt.title('Average Order Amount by Promotion Type (Top 5)')
    plt.xlabel('Promotion Type')
    plt.ylabel('Average Order Amount')
    plt.xticks(rotation=30, ha='right', fontsize=12)

    # 2. Conversion rate by promotion type (Top 5)
    plt.subplot(2, 2, 2)
    promotion_stats_top5['conversion_rate'].plot(kind='bar')
    plt.title('Conversion Rate by Promotion Type (Top 5)')
    plt.xlabel('Promotion Type')
    plt.ylabel('Conversion Rate')
    plt.xticks(rotation=30, ha='right', fontsize=12)

    # 3. Average order value by promotion type (Top 5)
    plt.subplot(2, 2, 3)
    promotion_stats_top5['average_order_value'].plot(kind='bar')
    plt.title('Average Order Value by Promotion Type (Top 5)')
    plt.xlabel('Promotion Type')
    plt.ylabel('Average Order Value')
    plt.xticks(rotation=30, ha='right', fontsize=12)

    # 4. Average discount amount by promotion type (Top 5)
    plt.subplot(2, 2, 4)
    promotion_stats_top5[('total_discount', 'mean')].plot(kind='bar')
    plt.title('Average Discount Amount by Promotion Type (Top 5)')
    plt.xlabel('Promotion Type')
    plt.ylabel('Average Discount Amount')
    plt.xticks(rotation=30, ha='right', fontsize=12)

    plt.tight_layout(pad=3.0)
    plt.savefig('promotion_analysis_top5.png')
    plt.close()

    #  促销组合one-hot编码，去掉一个基准组
    promo_dummies = pd.get_dummies(df['promotion_type'], drop_first=True)

    #  目标变量
    # 单笔订单的分析结果
    y = df['final_unit_price'].astype(float)

    #  合并特征
    X = promo_dummies.astype(float)
    X = sm.add_constant(X)

    #  建立回归模型
    model = sm.OLS(y, X).fit()
    print(model.summary())

    # 单笔订单回归分析
    coefs = model.params.drop('const')
    coefs.plot(kind='bar')
    plt.title('Effect of Promotion Types on Order Amount')
    plt.ylabel('Regression Coefficient')
    plt.tight_layout()
    plt.savefig('order_amount_regression.png')
    plt.close()

    # 分组统计：每种促销方式的总销售额
    df['order_sales'] = df['final_unit_price'] * df['quantity']
    sales_by_promo = df.groupby('promotion_type')['order_sales'].sum().sort_values(ascending=False)
    print('\nTotal Sales by Promotion Type:')
    print(sales_by_promo)
    sales_by_promo.to_csv('total_sales_by_promotion_type.csv')
    # 可视化Top 10
    sales_by_promo.head(10).plot(kind='bar')
    plt.title('Total Sales by Promotion Type (Top 10)')
    plt.xlabel('Promotion Type')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=30, ha='right', fontsize=12)
    plt.tight_layout()
    plt.savefig('total_sales_by_promotion_type.png')
    plt.close()

    # 计算每行订单的销售额和促销让利金额
    df['order_discount'] = df['total_discount'] * df['quantity']

    # 分组统计每种促销方式的总销售额和总让利金额
    roi_stats = df.groupby('promotion_type').agg(
        total_sales=('order_sales', 'sum'),
        total_discount=('order_discount', 'sum')
    )

    # 计算ROI
    # 填充0
    roi_stats['ROI'] = roi_stats['total_sales'] / roi_stats['total_discount'].replace(0, float('nan'))

    # 按ROI排序，输出Top 10
    roi_stats = roi_stats.sort_values('ROI', ascending=False)
    print('\nROI by Promotion Type (Top 10):')
    print(roi_stats[['total_sales', 'total_discount', 'ROI']].head(10))
    roi_stats.to_csv('promotion_roi_analysis.csv')

    # 可视化
    roi_stats['ROI'].head(10).plot(kind='bar')
    plt.title('ROI by Promotion Type (Top 10)')
    plt.xlabel('Promotion Type')
    plt.ylabel('ROI')
    plt.tight_layout()
    plt.savefig('promotion_roi_top10.png')
    plt.close()

    # 综合分析表：加权得分选出高销售额+高客单价+高转化率+高ROI的促销方式
    # 读取分组统计结果
    df1 = pd.read_csv('total_sales_by_promotion_type.csv', index_col=0)
    df2 = pd.read_csv('promotion_effectiveness_analysis.csv', index_col=0)
    df_roi = pd.read_csv('promotion_roi_analysis.csv', index_col=0)
    df_score = df1.join(df2[['final_unit_price', 'conversion_rate']]).join(df_roi[['ROI']])
    df_score = df_score.rename(columns={'order_sales': 'total_sales', 'final_unit_price': 'avg_order_amount'})

    # 归一化
    scaler = MinMaxScaler()
    df_score[['total_sales_norm', 'avg_order_amount_norm', 'conversion_rate_norm', 'ROI_norm']] = scaler.fit_transform(
        df_score[['total_sales', 'avg_order_amount', 'conversion_rate', 'ROI']]
    )

    # 综合得分（四项均分权重）
    df_score['score'] = (
        df_score['total_sales_norm'] * 0.25 +
        df_score['avg_order_amount_norm'] * 0.25 +
        df_score['conversion_rate_norm'] * 0.25 +
        df_score['ROI_norm'] * 0.25
    )

    # 排序输出Top 10
    df_score = df_score.sort_values('score', ascending=False)
    print('\n综合加权得分Top 10促销方式（含ROI）:')
    print(df_score[['total_sales', 'avg_order_amount', 'conversion_rate', 'ROI', 'score']].head(10))
    df_score.to_csv('promotion_composite_score_with_roi.csv')

    # ARIMA时间序列建模与预测
    time_series_analysis(df)

def time_series_analysis(df):
    # 1. 按日期汇总销售额
    df['order_date'] = pd.to_datetime(df['order_date'])
    daily_sales = df.groupby('order_date')['order_sales'].sum()
    
    # 2. 检查数据平稳性
    adf_result = adfuller(daily_sales)
    print("\n=== 时间序列平稳性检验 ===")
    print(f'ADF Statistic: {adf_result[0]}')
    print(f'p-value: {adf_result[1]}')
    
    #  使用ARIMA模型
    model = ARIMA(daily_sales, order=(1,1,1)) 
    model_fit = model.fit()
    
    print("\n=== ARIMA模型参数 ===")
    print(model_fit.summary())
    
    
    forecast = model_fit.forecast(steps=30)
    
    # 创建预测日期索引
    last_date = daily_sales.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
    forecast_series = pd.Series(forecast, index=forecast_dates)
    
    #  合并实际与预测，画图
    plt.figure(figsize=(12,6))
    plot_start = daily_sales.index.max() - pd.Timedelta(days=89)
    plt.plot(daily_sales[daily_sales.index >= plot_start], label='Actual Sales')
    plt.plot(forecast_series, label='ARIMA Forecast')
    plt.title('Sales Trend and ARIMA Forecast')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.legend()
    plt.tight_layout()
    plt.savefig('arima_sales_forecast.png')
    plt.show()
    plt.close()
    
    #  计算预测误差
    mae = mean_absolute_error(daily_sales[-30:], forecast_series)
    rmse = np.sqrt(mean_squared_error(daily_sales[-30:], forecast_series))
    
    print("\n=== 预测误差 ===")
    print(f'MAE: {mae:.2f}')
    print(f'RMSE: {rmse:.2f}')

if __name__ == "__main__":
    main() 